from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import stat
import struct
from unittest.mock import patch
import zipfile

from jsonschema import Draft202012Validator
import pytest

from teaching_skill_miner import teacher_agent_data_rights as data_rights
from teaching_skill_miner import teacher_agent_dashboard as dashboard_module
from teaching_skill_miner import teacher_agent_store as store_module
from teaching_skill_miner.cli import (
    _read_backup_passphrase,
    _read_curriculum_scope_key,
    _read_learner_key_secret,
    main as cli_main,
)
from teaching_skill_miner.student_model import (
    initialize_student_model,
    update_student_model,
)
from teaching_skill_miner.teacher_agent import (
    _refresh_integrity,
    advance_teacher_agent_session as real_advance_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_data_rights import (
    TeacherAgentDataRightsError,
    create_encrypted_local_backup,
    deletion_confirmation,
    restore_encrypted_local_backup,
    validate_project_export_archive,
)
from teaching_skill_miner.teacher_agent_syllabus import (
    _seal_generated_syllabus,
    syllabus_lesson_start_payload,
)
from teaching_skill_miner.teacher_agent_task_registry import (
    BackgroundTaskNotFoundError,
    DurableBackgroundTaskRegistry,
)

ROOT = Path(__file__).resolve().parents[1]
LEARNER_SECRET = b"data-rights-learning-secret-material-32-bytes"


def _snapshot(tmp_path: Path):
    return build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library_v2.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        store_path=tmp_path / "sessions.jsonl",
        syllabus_store_path=tmp_path / "syllabi",
        project_store_path=tmp_path / "projects",
        resource_index_store_path=tmp_path / "resources",
    )


def _learning_snapshot(tmp_path: Path):
    return build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library_v2.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        store_path=tmp_path / "sessions.jsonl",
        syllabus_store_path=tmp_path / "syllabi",
        project_store_path=tmp_path / "projects",
        resource_index_store_path=tmp_path / "resources",
        learning_record_store_path=tmp_path / "learning.jsonl",
        learner_key_secret=LEARNER_SECRET,
        learner_tenant_id="data-rights-school",
    )


def _seed_learning_session(snapshot, *, start_key: str = "rights.start"):
    profile = dict(snapshot.demo_input["student_profile"])
    profile["profile_ref"] = "student-data-rights"
    session = snapshot.start(
        {
            "goal": snapshot.demo_input["goal"],
            "student_profile": profile,
            "start_idempotency_key": start_key,
        }
    )

    def authoritative_advance(value, **kwargs):
        advanced = real_advance_teacher_agent_session(value, **kwargs)
        kc_id = "kc_state_definition"
        model = initialize_student_model(
            advanced["student_state"].get("knowledge_mastery", {}),
            goal={
                "concept": "dynamic programming",
                "knowledge_components": [{"kc_id": kc_id, "label": "state definition"}],
                "knowledge_spec": {
                    "schema": "teaching_skill_miner.teacher_goal_knowledge_spec.v1",
                    "status": "teacher_provided",
                    "claim_boundary": {"authoritative_for_runtime_grading": True},
                },
            },
        )
        updated = update_student_model(
            model,
            signal="correct",
            confidence=0.9,
            focus_dimension="conceptual",
            knowledge_component_ids=[kc_id],
            answer_alignment="aligned",
            assessment_eligible=True,
            authoritative=True,
            round_number=int(advanced["round"]),
            evidence_id="evidence.rights.seed",
            item_id="item.rights.seed",
            question_id="question.rights.seed",
            rubric_id="rubric.rights.seed",
            observed_at="2000-01-01T00:00:01Z",
            time_basis="session_logical",
            source="validated_teacher_rubric",
        )
        advanced["student_state"]["student_model"] = updated
        return _refresh_integrity(advanced)

    step_body = {
        "session_id": session["session_id"],
        "expected_round": session["rounds_completed"],
        "expected_question_id": session["expected_question_id"],
        "expected_context_version": session["context_version"],
        "profile_revision": session["profile_summary"]["profile_revision"],
        "idempotency_key": "rights.seed.step",
        "learner_response": "authoritative rights answer",
        "signal": "correct",
        "signal_confidence": 0.9,
    }
    with patch(
        "teaching_skill_miner.teacher_agent_dashboard.advance_teacher_agent_session",
        side_effect=authoritative_advance,
    ):
        session = snapshot.step(step_body)
    record = snapshot.sessions[session["session_id"]]
    learner_key = snapshot._learner_key_for_record(record)
    assert learner_key is not None
    assert snapshot.learning_record_store.recover().event_count == 1
    return session, profile, learner_key


def _private_project(snapshot, title: str = "私有项目"):
    private_text = "私有讲义：先定义状态，再写出状态转移。"
    staged = snapshot.upload_resource(
        {
            "resource_idempotency_key": f"resource-{title}",
            "mime_type": "text/plain",
            "display_name": "私有讲义.txt",
            "data_base64": base64.b64encode(private_text.encode()).decode(),
        }
    )
    session = snapshot.start(
        {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": f"start-{title}",
            "staged_resource_ids": [staged["resource"]["staged_resource_id"]],
        }
    )
    project = snapshot.create_project({"title": title})["project"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "teaching_session", "reference_id": session["session_id"]},
    )
    assert snapshot.read_project(project["project_id"])["project"]["resource_ids"] == []
    return (
        project,
        session,
        session["teaching_resources"][0]["resource_id"],
        private_text,
    )


def _background_task_project(snapshot, title: str = "后台教学项目"):
    request_marker = f"private-background-request-{title}"
    record, _cursor = snapshot.open_harness_stream(
        {
            "operation": "start",
            "request_id": f"background-rights-{title}",
            "payload": {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": request_marker,
            },
        }
    )
    assert record.handle.wait(timeout=5)["status"] == "completed"
    assert record.task_id is not None
    assert record.session_id is not None
    project = snapshot.create_project({"title": title})["project"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "teaching_session", "reference_id": record.session_id},
    )
    return project, record, request_marker


def _session_only_syllabus_project(snapshot, title: str = "仅会话引用大纲"):
    source_text = "仅由大纲来源图引用的私有材料。"
    staged = snapshot.upload_resource(
        {
            "resource_idempotency_key": f"syllabus-source-{title}",
            "mime_type": "text/plain",
            "display_name": "大纲来源.txt",
            "data_base64": base64.b64encode(source_text.encode()).decode(),
        }
    )
    resource_id = staged["resource"]["resource_id"]
    syllabus = _seal_generated_syllabus(
        {
            "title": "递推关系入门",
            "description": "从最小例子理解递推关系。",
            "audience": "零基础学习者",
            "estimated_duration_minutes": 30,
            "learning_objectives": ["识别递推关系的组成部分"],
            "prerequisites": ["能读懂简单数列"],
            "modules": [
                {
                    "title": "递推基础",
                    "description": "识别初始条件与递推规则。",
                    "lessons": [
                        {
                            "title": "初始条件和递推规则",
                            "objective": "能区分初始条件与递推规则。",
                            "summary": "通过数列例子建立两个核心概念。",
                            "duration_minutes": 30,
                            "knowledge_components": ["初始条件", "递推规则"],
                            "materials": {
                                "example": "观察 1, 1, 2, 3, 5 的生成过程。",
                                "practice": "标出给定递推式中的初始条件。",
                                "transfer_task": "为台阶问题写出递推关系。",
                            },
                        }
                    ],
                }
            ],
        },
        model="local-test-model",
        source_resource_ids=[resource_id],
        created_at="2026-08-12T00:00:00Z",
    )
    snapshot.syllabus_store.save(syllabus)
    lesson = syllabus_lesson_start_payload(syllabus, "lesson_01_01")
    session = snapshot.start(
        {
            "goal": lesson["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": f"syllabus-start-{title}",
        }
    )
    assert session["teaching_resources"] == []
    project = snapshot.create_project({"title": title})["project"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "teaching_session", "reference_id": session["session_id"]},
    )
    stored = snapshot.read_project(project["project_id"])["project"]
    assert stored["syllabus_ids"] == []
    assert stored["resource_ids"] == []
    return project, session, syllabus, resource_id


def _trash(snapshot, project):
    return snapshot.trash_project(project["project_id"], {})


def _purge(snapshot, project, trash):
    return snapshot.purge_project(
        project["project_id"],
        {
            "recovery_token": trash["recovery_token"],
            "confirmation": deletion_confirmation(project["project_id"]),
        },
    )


def test_export_traverses_historical_session_resources_and_verifies_manifest(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, resource_id, private_text = _private_project(snapshot)
    archive = snapshot.export_project(project["project_id"])
    manifest = validate_project_export_archive(archive.payload)
    assert manifest["claim_boundary"] == {
        "contains_private_content": True,
        "built_from_public_projection": False,
        "suitable_for_public_release": False,
    }
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        names = set(exported.namelist())
        assert f"sessions/{session['session_id']}.json" in names
        assert f"resources/{resource_id}.json" in names
        assert private_text in exported.read(f"resources/{resource_id}.json").decode()
        assert (
            exported.read("manifest.sha256")
            .decode()
            .startswith(archive.manifest_sha256)
        )


def test_background_task_private_recovery_is_exported_then_purged(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    project, record, request_marker = _background_task_project(snapshot)
    task_id = str(record.task_id)

    archive = snapshot.export_project(project["project_id"])
    validate_project_export_archive(archive.payload)
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        task_path = f"background_tasks/{task_id}.json"
        assert task_path in exported.namelist()
        task_export = exported.read(task_path).decode("utf-8")
        assert request_marker in task_export

    receipt = _purge(snapshot, project, _trash(snapshot, project))["deletion_receipt"]
    assert receipt["deleted_counts"]["background_tasks"] == 1
    assert receipt["deleted_counts"]["background_task_tombstones"] == 1
    with pytest.raises(BackgroundTaskNotFoundError):
        snapshot.task_registry.get_private(task_id)
    registry_text = snapshot.task_registry.path.read_text(encoding="utf-8")
    assert request_marker not in registry_text
    assert task_id in registry_text
    assert task_id not in json.dumps(receipt)


def test_shared_background_task_is_retained_for_other_project(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    first, record, request_marker = _background_task_project(snapshot, "后台项目甲")
    second = snapshot.create_project({"title": "后台项目乙"})["project"]
    snapshot.add_project_reference(
        second["project_id"],
        {"kind": "teaching_session", "reference_id": record.session_id},
    )

    receipt = _purge(snapshot, first, _trash(snapshot, first))["deletion_receipt"]
    assert receipt["deleted_counts"]["background_tasks"] == 0
    assert receipt["retained_shared_counts"]["background_tasks"] == 1
    retained = snapshot.task_registry.get_private(str(record.task_id))
    assert request_marker in json.dumps(retained, ensure_ascii=False)


def test_learning_record_is_exported_and_exclusive_project_purge_is_fenced(
    tmp_path: Path,
) -> None:
    snapshot = _learning_snapshot(tmp_path)
    session, _profile, learner_key = _seed_learning_session(snapshot)
    project = snapshot.create_project({"title": "学习记录项目"})["project"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "teaching_session", "reference_id": session["session_id"]},
    )

    archive = snapshot.export_project(project["project_id"])
    validate_project_export_archive(archive.payload)
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        learning_path = f"learning_records/{learner_key}.json"
        assert learning_path in exported.namelist()
        learning_export = json.loads(exported.read(learning_path))
        assert learning_export["record"]["learner_key"] == learner_key
        assert len(learning_export["store_events"]) == 1

    secret_path = tmp_path / "learner.secret"
    secret_path.write_bytes(LEARNER_SECRET)
    secret_path.chmod(0o600)
    receipt = _purge(snapshot, project, _trash(snapshot, project))["deletion_receipt"]
    assert receipt["deleted_counts"]["learning_records"] == 1
    assert receipt["deleted_counts"]["learning_record_events"] == 1
    assert learner_key not in json.dumps(receipt)
    assert snapshot.learning_record_store.get_learner_record(learner_key) is None
    tombstones = json.loads(
        snapshot.learning_record_store.tombstone_path.read_text(encoding="utf-8")
    )
    assert learner_key in tombstones["tombstones"]
    assert secret_path.read_bytes() == LEARNER_SECRET


def test_project_purge_preserves_learning_record_used_by_unprojected_session(
    tmp_path: Path,
) -> None:
    snapshot = _learning_snapshot(tmp_path)
    first, profile, learner_key = _seed_learning_session(snapshot)
    second = snapshot.start(
        {
            "goal": snapshot.demo_input["goal"],
            "student_profile": profile,
            "start_idempotency_key": "rights.start.second",
        }
    )
    assert (
        snapshot._learner_key_for_record(snapshot.sessions[second["session_id"]])
        == learner_key
    )
    project = snapshot.create_project({"title": "局部项目"})["project"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "teaching_session", "reference_id": first["session_id"]},
    )

    receipt = _purge(snapshot, project, _trash(snapshot, project))["deletion_receipt"]
    assert receipt["deleted_counts"]["learning_records"] == 0
    assert receipt["deleted_counts"]["learning_source_observations_suspended"] == 1
    assert receipt["retained_shared_counts"]["learning_records"] == 1
    retained = snapshot.learning_record_store.get_learner_record(learner_key)
    assert retained is not None
    component = next(
        iter(next(iter(retained["knowledge_components"].values())).values())
    )
    assert component["schedule"]["state"] == "suspended"
    assert component["schedule"]["suspension_reason"] == "source_deleted"
    assert snapshot.store.recover_session(second["session_id"]) is not None


def test_export_validator_rejects_compressed_member_size_mismatch_before_read() -> None:
    member = b"A" * 100_000
    manifest = {
        "schema": data_rights.PROJECT_EXPORT_SCHEMA,
        "project_id": "project_" + "1" * 24,
        "exported_at": "2026-08-12T00:00:00Z",
        "entry_count": 1,
        "entries": [
            {
                "path": "resources/res.json",
                "bytes": 1,
                "sha256": "0" * 64,
                "media_type": "application/json",
                "data_class": "teaching_resource_private_index",
                "reference_id": "res",
            }
        ],
        "claim_boundary": {
            "contains_private_content": True,
            "built_from_public_projection": False,
            "suitable_for_public_release": False,
        },
    }
    material = data_rights._json_document(manifest)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("resources/res.json", member)
        archive.writestr("manifest.json", material)
        archive.writestr(
            "manifest.sha256",
            f"{data_rights.sha256(material).hexdigest()}  manifest.json\n",
        )
    with pytest.raises(TeacherAgentDataRightsError, match="ZIP entry size"):
        validate_project_export_archive(output.getvalue())


def test_exclusive_purge_is_complete_content_free_and_idempotent(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, resource_id, private_text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    first = _purge(snapshot, project, trash)
    assert first == _purge(snapshot, project, trash)
    receipt = first["deletion_receipt"]
    assert receipt["deleted_counts"]["sessions"] == 1
    assert receipt["deleted_counts"]["resources"] == 1
    assert receipt["content_retained_in_receipt"] is False
    assert receipt["user_managed_export_copies_deleted"] is False
    assert receipt["backup_scope"] == "configured_live_stores_only"
    encoded = json.dumps(receipt, ensure_ascii=False)
    assert session["session_id"] not in encoded
    assert resource_id not in encoded
    assert private_text not in encoded
    assert snapshot.store.recover_session(session["session_id"]) is None
    assert snapshot.resource_index_store.get(resource_id) is None
    assert snapshot.list_trashed_projects()["projects"] == []


def test_exclusive_session_lock_is_held_through_irreversible_store_commit(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, _resource, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    record = snapshot.sessions[session["session_id"]]
    original_purge = snapshot.store.purge_sessions

    def assert_fenced(session_ids):
        assert record.lock.acquire(blocking=False) is False
        return original_purge(session_ids)

    monkeypatch.setattr(snapshot.store, "purge_sessions", assert_fenced)
    assert (
        _purge(snapshot, project, trash)["deletion_receipt"]["storage_cleanup_state"]
        == "completed"
    )


def test_shared_session_and_transitive_resource_are_never_deleted(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    first, session, resource_id, _text = _private_project(snapshot, "项目甲")
    second = snapshot.create_project({"title": "项目乙"})["project"]
    snapshot.add_project_reference(
        second["project_id"],
        {"kind": "teaching_session", "reference_id": session["session_id"]},
    )
    receipt = _purge(snapshot, first, _trash(snapshot, first))["deletion_receipt"]
    assert receipt["deleted_counts"]["sessions"] == 0
    assert receipt["deleted_counts"]["resources"] == 0
    assert receipt["retained_shared_counts"]["sessions"] == 1
    assert receipt["retained_shared_counts"]["resources"] == 1
    assert snapshot.store.recover_session(session["session_id"]) is not None
    assert snapshot.resource_index_store.get(resource_id) is not None


def test_session_only_syllabus_source_graph_exports_and_remains_shared(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    first, session, syllabus, resource_id = _session_only_syllabus_project(snapshot)

    archive = snapshot.export_project(first["project_id"])
    validate_project_export_archive(archive.payload)
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        assert f"sessions/{session['session_id']}.json" in exported.namelist()
        assert f"syllabi/{syllabus['syllabus_id']}.json" in exported.namelist()
        family = snapshot.syllabus_version_store.read_family(syllabus["syllabus_id"])
        assert f"syllabus_versions/{family['family_id']}.json" in exported.namelist()
        assert f"resources/{resource_id}.json" in exported.namelist()

    second = snapshot.create_project({"title": "共享会话项目"})["project"]
    snapshot.add_project_reference(
        second["project_id"],
        {"kind": "teaching_session", "reference_id": session["session_id"]},
    )
    receipt = _purge(snapshot, first, _trash(snapshot, first))["deletion_receipt"]
    assert receipt["retained_shared_counts"] == {
        "sessions": 1,
        "syllabi": 1,
        "syllabus_version_families": 1,
        "resources": 1,
        "learning_records": 0,
        "adjudications": 0,
        "adjudication_evidence": 0,
        "consent_receipts": 0,
        "background_tasks": 0,
    }
    assert snapshot.store.recover_session(session["session_id"]) is not None
    assert snapshot.syllabus_store.read(syllabus["syllabus_id"])
    assert snapshot.resource_index_store.get(resource_id) is not None


def test_exclusive_syllabus_family_is_compacted_and_fenced_on_purge(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    project, _session, syllabus, _resource_id = _session_only_syllabus_project(
        snapshot, "独占大纲版本项目"
    )
    family = snapshot.syllabus_version_store.read_family(syllabus["syllabus_id"])
    receipt = _purge(snapshot, project, _trash(snapshot, project))["deletion_receipt"]

    assert receipt["deleted_counts"]["syllabus_version_families"] == 1
    assert receipt["deleted_counts"]["syllabus_version_events"] == 1
    state_text = snapshot.syllabus_version_store.path.read_text()
    assert family["family_id"] not in state_text
    assert syllabus["syllabus_id"] not in state_text
    with pytest.raises(Exception, match="permanently erased"):
        snapshot.syllabus_version_store.register(
            syllabus, idempotency_key="purged-family-resurrection"
        )


def test_memory_only_session_store_can_be_purged(tmp_path: Path) -> None:
    snapshot = build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library_v2.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        project_store_path=tmp_path / "projects",
        resource_index_store_path=tmp_path / "resources",
        syllabus_store_path=tmp_path / "syllabi",
    )
    project, session, resource_id, _text = _private_project(snapshot)
    assert snapshot.store is None
    receipt = _purge(snapshot, project, _trash(snapshot, project))["deletion_receipt"]
    assert receipt["deleted_counts"]["sessions"] == 1
    assert receipt["deleted_counts"]["session_store_events"] == 0
    assert snapshot.resource_index_store.get(resource_id) is None
    assert session["session_id"] not in snapshot.sessions


def test_confirmation_token_and_precommit_failure_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, resource_id, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    with pytest.raises(TeacherAgentDashboardError, match="confirmation"):
        snapshot.purge_project(
            project["project_id"],
            {"recovery_token": trash["recovery_token"], "confirmation": "DELETE"},
        )
    with pytest.raises(Exception, match="recovery_token"):
        snapshot.purge_project(
            project["project_id"],
            {
                "recovery_token": f"restore_project_{'0' * 24}_{'0' * 12}",
                "confirmation": deletion_confirmation(project["project_id"]),
            },
        )

    source = snapshot._learning_projects()._trash_identity(trash["recovery_token"])[1]
    original_replace = Path.replace
    fired = False

    def fail_project_stage(self: Path, target: Path):
        nonlocal fired
        if self == source and not fired:
            fired = True
            raise OSError("synthetic stage failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_project_stage)
    with pytest.raises(OSError, match="synthetic stage failure"):
        _purge(snapshot, project, trash)
    assert source.is_file()
    assert snapshot.store.recover_session(session["session_id"]) is not None
    assert snapshot.resource_index_store.get(resource_id) is not None
    assert not snapshot._deletion_receipt_path(trash["recovery_token"]).exists()


def test_postcommit_unlink_failure_hides_project_and_retry_only_finishes_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, _resource, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    original_unlink = Path.unlink
    fired = False

    def fail_once(self: Path, *args, **kwargs):
        nonlocal fired
        if ".purge-" in self.name and not fired:
            fired = True
            raise OSError("synthetic unlink failure")
        return original_unlink(self, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", fail_once)
        with pytest.raises(TeacherAgentDataRightsError, match="cleanup is pending"):
            _purge(snapshot, project, trash)
    assert snapshot.store.recover_session(session["session_id"]) is None
    assert snapshot.list_trashed_projects()["projects"] == []
    with pytest.raises(Exception, match="not found"):
        snapshot.restore_project(
            project["project_id"], {"recovery_token": trash["recovery_token"]}
        )
    assert (
        _purge(snapshot, project, trash)["deletion_receipt"]["storage_cleanup_state"]
        == "completed"
    )


def test_directory_fsync_fault_after_store_replace_is_committed_pending(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, _resource, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    original_fsync = store_module.os.fsync
    directory_fsync_calls = 0

    def fail_directory_fsync(descriptor: int):
        nonlocal directory_fsync_calls
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsync_calls += 1
            if directory_fsync_calls == 2:
                raise OSError("synthetic directory fsync failure")
        return original_fsync(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(store_module.os, "fsync", fail_directory_fsync)
        with pytest.raises(TeacherAgentDataRightsError, match="remains inaccessible"):
            _purge(snapshot, project, trash)
    assert snapshot.store.recover_session(session["session_id"]) is None
    assert snapshot.list_trashed_projects()["projects"] == []
    intent = json.loads(
        snapshot._deletion_receipt_path(trash["recovery_token"]).read_text()
    )
    assert intent["state"] == "committed_cleanup_pending"
    assert (
        _purge(snapshot, project, trash)["deletion_receipt"]["storage_cleanup_state"]
        == "completed"
    )


def test_prepared_intent_parent_fsync_precedes_target_rename(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, resource_id, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    original_fsync = store_module.os.fsync
    original_replace = Path.replace
    target_renames = 0

    def fail_receipt_directory_fsync(descriptor: int):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("synthetic prepared intent fsync failure")
        return original_fsync(descriptor)

    def count_target_rename(self: Path, target: Path):
        nonlocal target_renames
        if ".purge-" in target.name:
            target_renames += 1
        return original_replace(self, target)

    monkeypatch.setattr(store_module.os, "fsync", fail_receipt_directory_fsync)
    monkeypatch.setattr(Path, "replace", count_target_rename)
    with pytest.raises(OSError, match="prepared intent fsync failure"):
        _purge(snapshot, project, trash)
    assert target_renames == 0
    assert snapshot.store.recover_session(session["session_id"]) is not None
    assert snapshot.resource_index_store.get(resource_id) is not None
    assert (
        snapshot.list_trashed_projects()["projects"][0]["project_id"]
        == project["project_id"]
    )


def test_committed_intent_parent_fsync_failure_never_reexposes_session(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, _resource, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    original_fsync_directory = dashboard_module._fsync_stream_directory
    calls = 0

    def fail_after_store_commit(root: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic committed intent fsync failure")
        original_fsync_directory(root)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            dashboard_module, "_fsync_stream_directory", fail_after_store_commit
        )
        with pytest.raises(TeacherAgentDataRightsError, match="remains inaccessible"):
            _purge(snapshot, project, trash)

    assert snapshot.store.recover_session(session["session_id"]) is None
    assert session["session_id"] not in snapshot.sessions
    assert snapshot.list_trashed_projects()["projects"] == []
    completed = _purge(snapshot, project, trash)["deletion_receipt"]
    assert completed["storage_cleanup_state"] == "completed"


def test_crash_after_session_commit_is_reconciled_by_fresh_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    project, session, _resource, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    original_purge = snapshot.store.purge_sessions

    def commit_then_crash(session_ids):
        original_purge(session_ids)
        raise SystemExit("synthetic process crash after session-store commit")

    monkeypatch.setattr(snapshot.store, "purge_sessions", commit_then_crash)
    with pytest.raises(SystemExit, match="synthetic process crash"):
        _purge(snapshot, project, trash)
    assert snapshot.store.recover_session(session["session_id"]) is None

    restarted = _snapshot(tmp_path)
    completed = _purge(restarted, project, trash)["deletion_receipt"]
    assert completed["storage_cleanup_state"] == "completed"
    assert completed["deleted_counts"]["sessions"] == 1
    assert restarted.store.recover_session(session["session_id"]) is None
    assert restarted.list_trashed_projects()["projects"] == []


def test_backup_restore_and_hostile_header_zip_bounds(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "private.json").write_text('{"answer":"private"}\n')
    password = "correct horse battery staple"
    payload, receipt = create_encrypted_local_backup(
        {"projects": source}, passphrase=password
    )
    assert receipt["local_only"] is True
    assert receipt["cloud_backup_claimed"] is False
    restored = restore_encrypted_local_backup(
        payload, passphrase=password, destination=tmp_path / "drill"
    )
    assert restored["all_hashes_verified"] is True
    assert restored["live_store_overwritten"] is False
    with pytest.raises(TeacherAgentDataRightsError, match="authentication failed"):
        restore_encrypted_local_backup(
            payload[:-1] + bytes([payload[-1] ^ 1]),
            passphrase=password,
            destination=tmp_path / "tampered",
        )

    header = {
        "schema": data_rights.LOCAL_BACKUP_SCHEMA,
        "created_at": "2026-08-12T00:00:00Z",
        "cipher": "AES-256-GCM",
        "kdf": "scrypt-n32768-r8-p1",
        "salt_b64": base64.b64encode(b"s" * 16).decode(),
        "nonce_b64": base64.b64encode(b"n" * 12).decode(),
        "plaintext_bytes": data_rights._MAX_BACKUP_BYTES + 1,
        "plaintext_sha256": "0" * 64,
        "local_only": True,
    }
    material = data_rights.canonical_json_bytes(header)
    hostile = (
        data_rights._BACKUP_MAGIC + struct.pack(">I", len(material)) + material + b"x"
    )
    with pytest.raises(TeacherAgentDataRightsError, match="contract"):
        restore_encrypted_local_backup(
            hostile, passphrase=password, destination=tmp_path / "oversized"
        )

    malicious = io.BytesIO()
    with zipfile.ZipFile(malicious, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = {
            "schema": data_rights.LOCAL_BACKUP_MANIFEST_SCHEMA,
            "created_at": "2026-08-12T00:00:00Z",
            "file_count": 1,
            "entries": [{"path": "stores/x/a", "bytes": 1, "sha256": "0" * 64}],
        }
        archive.writestr("stores/x/a", b"A" * 100_000)
        archive.writestr("backup-manifest.json", data_rights._json_document(manifest))
    monkeypatch.setattr(
        data_rights,
        "_decrypt_local_backup",
        lambda _payload, *, passphrase: (
            malicious.getvalue(),
            {"created_at": "2026-08-12T00:00:00Z"},
        ),
    )
    with pytest.raises(TeacherAgentDataRightsError, match="ZIP entry size"):
        restore_encrypted_local_backup(
            b"synthetic", passphrase=password, destination=tmp_path / "zip-bomb"
        )


def test_backup_passphrase_file_rejects_links_public_modes_and_large_files(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "passphrase.txt"
    secret.write_text("correct horse battery staple\n", encoding="utf-8")
    secret.chmod(0o600)
    assert _read_backup_passphrase(secret) == "correct horse battery staple"

    secret.chmod(0o640)
    with pytest.raises(ValueError, match="private regular file"):
        _read_backup_passphrase(secret)
    secret.chmod(0o600)

    linked = tmp_path / "linked-passphrase.txt"
    linked.symlink_to(secret)
    with pytest.raises(ValueError, match="private regular file"):
        _read_backup_passphrase(linked)

    oversized = tmp_path / "oversized-passphrase.txt"
    oversized.write_bytes(b"x" * 4097)
    oversized.chmod(0o600)
    with pytest.raises(ValueError, match="bounded private regular file"):
        _read_backup_passphrase(oversized)


def test_learner_key_secret_file_rejects_links_public_modes_and_bad_sizes(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "learner.secret"
    secret.write_bytes(LEARNER_SECRET)
    secret.chmod(0o600)
    assert _read_learner_key_secret(secret) == LEARNER_SECRET

    secret.chmod(0o640)
    with pytest.raises(ValueError, match="private regular file"):
        _read_learner_key_secret(secret)
    secret.chmod(0o600)

    linked = tmp_path / "linked-learner.secret"
    linked.symlink_to(secret)
    with pytest.raises(ValueError, match="private regular file"):
        _read_learner_key_secret(linked)

    short = tmp_path / "short-learner.secret"
    short.write_bytes(b"x" * 31)
    short.chmod(0o600)
    with pytest.raises(ValueError, match="bounded private regular file"):
        _read_learner_key_secret(short)


def test_curriculum_signing_key_cli_rotates_and_revokes_without_private_output(
    tmp_path: Path, capsys
) -> None:
    private_root = tmp_path / "worker-private"
    private_root.mkdir(mode=0o700)
    scope_key = tmp_path / "scope.key"
    scope_key.write_bytes(b"s" * 32)
    scope_key.chmod(0o600)
    assert _read_curriculum_scope_key(scope_key) == b"s" * 32

    def run(action: str, key_id: str | None = None) -> dict:
        arguments = [
            "teacher-agent-curriculum-key",
            action,
            "--private-root",
            str(private_root),
            "--scope-key-file",
            str(scope_key),
        ]
        if key_id is not None:
            arguments.extend(["--key-id", key_id])
        assert cli_main(arguments) == 0
        return json.loads(capsys.readouterr().out)

    initial = run("status")
    first = initial["active_key_id"]
    rotated = run("rotate")
    second = rotated["active_key_id"]
    assert first != second
    assert {row["status"] for row in rotated["keys"]} == {
        "active",
        "verification_only",
    }
    revoked = run("revoke", first)
    assert next(row for row in revoked["keys"] if row["key_id"] == first)[
        "status"
    ] == "revoked"
    serialized = json.dumps(revoked)
    assert "ciphertext" not in serialized
    assert "nonce" not in serialized
    assert "PRIVATE KEY" not in serialized
    with pytest.raises(ValueError, match="private regular file"):
        scope_key.chmod(0o640)
        _read_curriculum_scope_key(scope_key)


def test_backup_cli_round_trip_is_local_private_and_does_not_print_passphrase(
    tmp_path: Path, capsys
) -> None:
    project_store = tmp_path / "projects"
    project_store.mkdir()
    (project_store / "project.json").write_text(
        '{"private":"learner content"}\n', encoding="utf-8"
    )
    passphrase = "private backup passphrase"
    passphrase_file = tmp_path / "passphrase.txt"
    passphrase_file.write_text(passphrase + "\n", encoding="utf-8")
    passphrase_file.chmod(0o600)
    learning_store = tmp_path / "learning.jsonl"
    learning_store.write_text('{"private":"schedule"}\n', encoding="utf-8")
    learning_tombstones = tmp_path / ".learning.jsonl.erasure_tombstones.json"
    learning_tombstones.write_text('{"private":"erasure-fence"}\n', encoding="utf-8")
    learning_lock = tmp_path / ".learning.jsonl.lock"
    learning_lock.write_bytes(b"")
    learner_secret = tmp_path / "learner.secret"
    learner_secret.write_bytes(LEARNER_SECRET)
    learner_secret.chmod(0o600)
    consent_store = tmp_path / "remote-consent.json"
    consent_store.write_text('{"private":"signed consent audit"}\n', encoding="utf-8")
    consent_secret = tmp_path / "remote-consent.secret"
    consent_secret.write_bytes(b"backup-consent-signing-secret-material-32-bytes")
    consent_secret.chmod(0o600)
    session_store = tmp_path / "sessions.jsonl"
    session_store.write_text('{"private":"session"}\n', encoding="utf-8")
    syllabus_store = tmp_path / "syllabi"
    syllabus_store.mkdir()
    keyring_sentinel = b'{"private_key_ciphertext_base64":"backup-key-envelope"}\n'
    authority_sentinel = b'{"schema":"curriculum-authority-backup-sentinel"}\n'
    (syllabus_store / ".curriculum_signing_keyring.json").write_bytes(
        keyring_sentinel
    )
    (syllabus_store / ".curriculum_authority.json").write_bytes(
        authority_sentinel
    )
    task_registry = DurableBackgroundTaskRegistry(
        session_store.with_name(session_store.name + ".harness_streams")
    )
    retained_task, _ = task_registry.register(
        run_id="stream_" + "1" * 40,
        turn_id="turn_" + "2" * 40,
        operation="chat",
        request_fingerprint="3" * 64,
        private_request={
            "operation": "chat",
            "payload": {"messages": [{"role": "user", "content": "backup secret"}]},
            "request_id": "backup-task-retained",
            "run_id": "stream_" + "1" * 40,
            "turn_id": "turn_" + "2" * 40,
            "after_sequence": 0,
        },
        scope={"project_id": "project_" + "4" * 24},
        session_id=None,
    )
    purged_task, _ = task_registry.register(
        run_id="stream_" + "5" * 40,
        turn_id="turn_" + "6" * 40,
        operation="start",
        request_fingerprint="7" * 64,
        private_request={
            "operation": "start",
            "payload": {"start_idempotency_key": "purged backup secret"},
            "request_id": "backup-task-purged",
            "run_id": "stream_" + "5" * 40,
            "turn_id": "turn_" + "6" * 40,
            "after_sequence": 0,
        },
        scope={"project_id": "project_" + "8" * 24},
        session_id=None,
    )
    task_registry.mark_terminal(
        str(purged_task["task_id"]),
        status="completed",
        terminal_type="run.completed",
        last_sequence=1,
    )
    task_registry.purge_tasks([str(purged_task["task_id"])])
    backup = tmp_path / "backup.tsm"
    assert (
        cli_main(
            [
                "teacher-agent-backup",
                "--output",
                str(backup),
                "--passphrase-file",
                str(passphrase_file),
                "--session-store",
                str(session_store),
                "--syllabus-store",
                str(syllabus_store),
                "--project-store",
                str(project_store),
                "--resource-index-store",
                str(tmp_path / "missing-resources"),
                "--learning-record-store",
                str(learning_store),
                "--learner-key-secret-file",
                str(learner_secret),
                "--consent-store",
                str(consent_store),
                "--consent-signing-secret-file",
                str(consent_secret),
            ]
        )
        == 0
    )
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    destination = tmp_path / "restore-drill"
    assert (
        cli_main(
            [
                "teacher-agent-restore-drill",
                "--backup",
                str(backup),
                "--passphrase-file",
                str(passphrase_file),
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert passphrase not in output.out
    assert passphrase not in output.err
    restored = destination / "restored" / "stores" / "projects" / "project.json"
    assert json.loads(restored.read_text()) == {"private": "learner content"}
    restored_root = destination / "restored" / "stores"
    assert (
        restored_root / "syllabi" / ".curriculum_signing_keyring.json"
    ).read_bytes() == keyring_sentinel
    assert (
        restored_root / "syllabi" / ".curriculum_authority.json"
    ).read_bytes() == authority_sentinel
    assert (
        restored_root / "learning_records" / learning_store.name
    ).read_bytes() == learning_store.read_bytes()
    assert (
        restored_root / "learning_erasure_tombstones" / learning_tombstones.name
    ).read_bytes() == learning_tombstones.read_bytes()
    assert (
        restored_root / "learning_process_lock" / learning_lock.name
    ).read_bytes() == learning_lock.read_bytes()
    restored_secret = restored_root / "learner_key_secret" / learner_secret.name
    assert restored_secret.read_bytes() == LEARNER_SECRET
    assert stat.S_IMODE(restored_secret.stat().st_mode) == 0o600
    assert (
        restored_root / "remote_consent" / consent_store.name
    ).read_bytes() == consent_store.read_bytes()
    restored_consent_secret = (
        restored_root / "remote_consent_signing_secret" / consent_secret.name
    )
    assert restored_consent_secret.read_bytes() == consent_secret.read_bytes()
    assert stat.S_IMODE(restored_consent_secret.stat().st_mode) == 0o600
    restored_registry = json.loads(
        (
            restored_root
            / "harness_streams"
            / task_registry.path.relative_to(task_registry.root)
        ).read_text(encoding="utf-8")
    )
    assert str(retained_task["task_id"]) in restored_registry["tasks"]
    assert "backup secret" in json.dumps(restored_registry)
    assert str(purged_task["task_id"]) in restored_registry["purged_task_tombstones"]
    assert "purged backup secret" not in json.dumps(restored_registry)


def test_backup_cli_requires_learning_store_and_key_secret_together(
    tmp_path: Path,
) -> None:
    passphrase_file = tmp_path / "passphrase.txt"
    passphrase_file.write_text("correct horse battery staple\n", encoding="utf-8")
    passphrase_file.chmod(0o600)
    assert (
        cli_main(
            [
                "teacher-agent-backup",
                "--output",
                str(tmp_path / "backup.tsm"),
                "--passphrase-file",
                str(passphrase_file),
                "--learning-record-store",
                str(tmp_path / "learning.jsonl"),
            ]
        )
        == 1
    )


def test_schema_privacy_freshness_and_browser_cleanup_handle(tmp_path: Path) -> None:
    for name in (
        "learning_project_export_manifest.schema.json",
        "learning_project_deletion_receipt.schema.json",
    ):
        Draft202012Validator.check_schema(
            json.loads((ROOT / "schema" / name).read_text())
        )
    privacy = (ROOT / "PRIVACY.md").read_text()
    for marker in (
        "teachlab.chat.threads",
        "teachlab.remote-consent.v1",
        ".private/learning_projects/",
        ".harness_streams/",
        "Last storage-map review: 2026-08-12",
        "does not upload anywhere",
        "A safeguarding case is deliberately content-minimized but still sensitive",
        "queue entry alone is not proof that a human received it",
        "account deletion purges their live\n  store",
    ):
        assert marker in privacy
    snapshot = _snapshot(tmp_path)
    project, session, _resource, _text = _private_project(snapshot)
    trash = _trash(snapshot, project)
    assert trash["browser_session_handles_to_forget"] == [session["session_id"]]
    assert "learner_response" not in json.dumps(trash)
