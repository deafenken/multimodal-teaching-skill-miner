from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import io
from pathlib import Path
import zipfile

import pytest

from teaching_skill_miner.teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_SCHEMA,
    TeacherAuthorityVerifier,
    canonical_bytes,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_data_rights import (
    create_encrypted_local_backup,
    deletion_confirmation,
    restore_encrypted_local_backup,
)
from teaching_skill_miner.teacher_agent_resources import (
    extract_teaching_resource,
    validate_teaching_resources,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
AUTHORITY_KEY = b"r" * 32
SCOPE_ID = "scope_" + "a" * 48


def _blocked_resource(data: bytes, mime_type: str, **options) -> dict:
    resource = extract_teaching_resource(
        data,
        mime_type,
        display_name=str(options["display_name"]),
    )
    text = (
        "[内容冲突待确认：正文写成三个步骤，讲者备注写成四个步骤]\n"
        "正文：过程包含三个步骤。\n讲者备注：过程包含四个步骤。"
    )
    resource.update(
        {
            "extracted_text": text,
            "extracted_char_count": len(text),
            "original_extracted_char_count": len(text),
            "truncated": False,
            "needs_review": True,
            "requires_confirmation": True,
            "evidence_contract": {
                "schema": "teaching_skill_miner.resource_evidence_layers.v1",
                "layers": [
                    {
                        "layer_id": "layer_001",
                        "kind": "text_transcription",
                        "evidence_locator": "resource/extracted-text",
                        "status": "candidate",
                        "semantic_understanding_established": False,
                        "visual_verification_status": "not_verified",
                        "grading_evidence_allowed": False,
                        "mastery_evidence_allowed": False,
                    },
                    {
                        "layer_id": "layer_002",
                        "kind": "speaker_notes",
                        "evidence_locator": "resource/speaker-notes",
                        "status": "candidate",
                        "semantic_understanding_established": False,
                        "visual_verification_status": "not_verified",
                        "grading_evidence_allowed": False,
                        "mastery_evidence_allowed": False,
                    },
                ],
                "conflicts": [
                    {
                        "conflict_id": "resource_conflict_001",
                        "kind": "slide_notes_disagreement",
                        "description": "正文写成三个步骤，讲者备注写成四个步骤",
                        "evidence_locators": ["slide/body", "slide/speaker-notes"],
                        "resolution_status": "unresolved",
                    }
                ],
                "decision": "requires_confirmation",
                "transcription_is_semantic_understanding": False,
                "semantic_analysis_is_answer_correctness": False,
                "grading_evidence_allowed": False,
                "mastery_evidence_allowed": False,
                "visual_verification_status": "not_applicable",
                "page_count_bound": None,
                "temporal_provenance": [],
            },
        }
    )
    validate_teaching_resources([resource])
    index_store = options.get("index_store")
    if index_store is not None:
        index_store.put(resource, indexed_text=text)
    return resource


def _snapshot(tmp_path: Path, *, authenticated: bool = True):
    verifier = (
        TeacherAuthorityVerifier(
            key=AUTHORITY_KEY,
            scope_id=SCOPE_ID,
            scope_key_version="k1",
            replay_store_path=tmp_path / "authority-replay.jsonl",
            clock=lambda: NOW,
        )
        if authenticated
        else None
    )
    return build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library_v2.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        resource_extractor=_blocked_resource,
        store_path=tmp_path / "sessions.jsonl",
        project_store_path=tmp_path / "projects",
        syllabus_store_path=tmp_path / "syllabi",
        resource_index_store_path=tmp_path / "resources",
        resource_review_store_path=tmp_path / "resource-reviews",
        teacher_authority_verifier=verifier,
    )


def _signature(value: dict) -> str:
    return (
        base64.urlsafe_b64encode(
            hmac.new(AUTHORITY_KEY, canonical_bytes(value), sha256).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )


def _signed(body: dict, *, nonce: str = "b" * 48) -> dict:
    envelope = {
        "schema": TEACHER_AUTHORITY_SCHEMA,
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "assurance": AUTHORITY_ASSURANCE,
        "scope_id": SCOPE_ID,
        "scope_key_version": "k1",
        "actor_principal_sha256": "1" * 64,
        "roles_sha256": "2" * 64,
        "role_policy_sha256": "3" * 64,
        "method": "POST",
        "path": "api/resource/review",
        "body_sha256": canonical_sha256(body),
        "idempotency_key_sha256": sha256(
            str(body["resource_review_idempotency_key"]).encode()
        ).hexdigest(),
        "issued_at": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "expires_at": (NOW + timedelta(minutes=2))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "nonce": "tan_" + nonce,
    }
    envelope["authority_id"] = "tauth_" + canonical_sha256(envelope)[:24]
    envelope["signature"] = _signature(envelope)
    return {**deepcopy(body), "_teacher_authority": envelope}


def _stage(snapshot) -> dict:
    return snapshot.upload_resource(
        {
            "resource_idempotency_key": "review-upload-key-001",
            "mime_type": "text/plain",
            "display_name": "跨层冲突讲义.txt",
            "data_base64": base64.b64encode(b"immutable original bytes").decode(),
        }
    )


def _review_body(staged: dict, *, version: int = 0, key: str = "review-key-0001"):
    resource = staged["resource"]
    return {
        "resource_id": resource["resource_id"],
        "staged_resource_id": resource["staged_resource_id"],
        "content_sha256": resource["content_sha256"],
        "expected_review_version": version,
        "resource_review_idempotency_key": key,
        "original_resource_sha256": resource["original_resource_sha256"],
        "reviewed_text": "教师对照原文件确认：过程包含三个步骤；讲者备注中的四是笔误。",
        "resolved_conflict_ids": ["resource_conflict_001"],
        "excluded_layer_ids": ["layer_002"],
        "attestations": {
            "compared_with_original_source": True,
            "uncertainties_removed_or_explicit": True,
            "not_an_answer_key": True,
            "context_only": True,
        },
        "review_note": "已对照原始文档逐项核对。",
    }


def test_review_is_authenticated_blocked_then_usable_and_restart_safe(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    staged = _stage(snapshot)
    stage_id = staged["resource"]["staged_resource_id"]
    assert staged["session_use"]["status"] == "blocked_pending_confirmation"
    assert (
        snapshot.bootstrap()["interaction_contract"][
            "chat_resource_reviewed_projection_endpoint_available"
        ]
        is True
    )

    with pytest.raises(TeacherAgentDashboardError, match="unresolved"):
        snapshot.start(
            {
                    "goal": snapshot.bootstrap()["default_goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "blocked-start-key-001",
                "staged_resource_ids": [stage_id],
            }
        )

    review = snapshot.review_resource(_signed(_review_body(staged)))
    assert review["session_use"]["status"] == "eligible_untrusted_context"
    assert review["review_scope"] == "untrusted_teaching_context_only"
    assert review["semantic_understanding_established"] is False
    assert review["grading_evidence_allowed"] is False
    assert review["mastery_evidence_allowed"] is False
    original = snapshot.resource_index_store.get_by_content_hash(
        staged["resource"]["content_sha256"]
    )
    assert original is not None and original["requires_confirmation"] is True
    assert "教师对照原文件" not in original["extracted_text"]

    second_body = _review_body(staged, version=1, key="review-key-0002")
    second_body["reviewed_text"] = (
        "教师再次对照原文件确认：过程确实包含三个步骤，第四项仅为备注笔误。"
    )
    second = snapshot.review_resource(_signed(second_body, nonce="c" * 48))
    assert second["resource"]["resource_review"]["review_version"] == 2
    assert second["resource"]["review_requirements"][
        "required_resolved_conflict_ids"
    ] == ["resource_conflict_001"]
    assert len(second["resource"]["review_requirements"]["layers"]) == 2

    started = snapshot.start(
        {
                "goal": snapshot.bootstrap()["default_goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "reviewed-start-key-001",
            "staged_resource_ids": [stage_id],
        }
    )
    record = snapshot.sessions[started["session_id"]]
    assert (
        "教师再次对照原文件"
        in record.session["teaching_resources"][0]["extracted_text"]
    )
    assert record.session["teaching_resources"][0]["grading_evidence_allowed"] is False
    assert record.session["teaching_resources"][0]["mastery_evidence_allowed"] is False

    restarted = _snapshot(tmp_path)
    restored = restarted._resolve_staged_resources([stage_id])[0]
    assert "教师再次对照原文件" in restored["extracted_text"]
    assert restored["review_projection"]["receipt"]["review_version"] == 2


def test_local_mode_and_browser_authority_or_answer_key_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    local = _snapshot(tmp_path / "local", authenticated=False)
    local_staged = _stage(local)
    with pytest.raises(TeacherAgentDashboardError, match="authenticated Apps API"):
        local.review_resource(_review_body(local_staged))

    snapshot = _snapshot(tmp_path / "authenticated")
    staged = _stage(snapshot)
    for index, attack in enumerate(
        [
            {"raw_media": "base64"},
            {"answer_key": {"correct": True}},
            {"grading_evidence_allowed": True},
            {"mastery_evidence_allowed": True},
            {"actor": {"role": "teacher"}},
            {"authority_receipt": {"forged": True}},
        ],
        1,
    ):
        body = {**_review_body(staged, key=f"review-attack-{index:04d}"), **attack}
        with pytest.raises(TeacherAgentDashboardError, match="fields are invalid"):
            snapshot.review_resource(_signed(body, nonce=f"{index:048x}"))
    assert (
        snapshot.resource_review_store.get(staged["resource"]["content_sha256"]) is None
    )


def test_authority_nonce_is_consumed_before_review_store_mutation(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    staged = _stage(snapshot)
    invalid = _review_body(staged, key="review-consume-before-mutate")
    invalid["attestations"] = {
        **invalid["attestations"],
        "context_only": False,
    }
    signed = _signed(invalid, nonce="d" * 48)
    with pytest.raises(TeacherAgentDashboardError, match="attestations"):
        snapshot.review_resource(signed)
    assert (
        snapshot.resource_review_store.get(staged["resource"]["content_sha256"]) is None
    )
    with pytest.raises(TeacherAgentDashboardError, match="authority was rejected"):
        snapshot.review_resource(signed)


def test_reviewed_projection_unblocks_chat_local_retrieval_without_raw_media(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    staged = _stage(snapshot)
    project = snapshot.create_project({"title": "复核资源项目"})["project"]
    project = snapshot.add_project_reference(
        project["project_id"],
        {"kind": "resource", "reference_id": staged["resource"]["resource_id"]},
    )["project"]
    refs = [
        {
            "resource_id": staged["resource"]["resource_id"],
            "staged_resource_id": staged["resource"]["staged_resource_id"],
        }
    ]
    with pytest.raises(TeacherAgentDashboardError, match="blocked pending"):
        snapshot._chat_resource_context(
            project=project,
            query="请总结附件中的步骤",
            resource_refs=refs,
        )
    snapshot.review_resource(_signed(_review_body(staged)))
    appendix, receipt = snapshot._chat_resource_context(
        project=project,
        query="请总结附件中的步骤",
        resource_refs=refs,
    )
    assert "教师对照原文件" in appendix
    assert receipt is not None
    assert receipt["raw_media_sent"] is False
    assert receipt["full_resource_text_sent"] is False
    assert receipt["context_role"] == "untrusted_quote_only"
    assert receipt["learner_or_mastery_evidence"] is False


def test_review_export_and_shared_ownership_purge_follow_original_resource(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    staged = _stage(snapshot)
    snapshot.review_resource(_signed(_review_body(staged)))
    resource_id = staged["resource"]["resource_id"]
    content_hash = staged["resource"]["content_sha256"]
    first = snapshot.create_project({"title": "项目甲"})["project"]
    first = snapshot.add_project_reference(
        first["project_id"], {"kind": "resource", "reference_id": resource_id}
    )["project"]
    second = snapshot.create_project({"title": "项目乙"})["project"]
    snapshot.add_project_reference(
        second["project_id"], {"kind": "resource", "reference_id": resource_id}
    )

    archive = snapshot.export_project(first["project_id"])
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        assert f"resources/{resource_id}.json" in exported.namelist()
        assert f"resource_reviews/{resource_id}.json" in exported.namelist()

    trashed = snapshot.trash_project(first["project_id"], {})
    retained = snapshot.purge_project(
        first["project_id"],
        {
            "recovery_token": trashed["recovery_token"],
            "confirmation": deletion_confirmation(first["project_id"]),
        },
    )["deletion_receipt"]
    assert retained["deleted_counts"]["resource_reviews"] == 0
    assert retained["retained_shared_counts"]["resource_reviews"] == 1
    assert snapshot.resource_review_store.get(content_hash) is not None

    second_current = snapshot.read_project(second["project_id"])["project"]
    trashed_second = snapshot.trash_project(second_current["project_id"], {})
    deleted = snapshot.purge_project(
        second_current["project_id"],
        {
            "recovery_token": trashed_second["recovery_token"],
            "confirmation": deletion_confirmation(second_current["project_id"]),
        },
    )["deletion_receipt"]
    assert deleted["deleted_counts"]["resource_reviews"] == 1
    assert snapshot.resource_review_store.get(content_hash) is None


def test_review_store_is_included_in_encrypted_local_backup(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    staged = _stage(snapshot)
    snapshot.review_resource(_signed(_review_body(staged)))
    content_hash = staged["resource"]["content_sha256"]

    payload, receipt = create_encrypted_local_backup(
        {"resource_reviews": snapshot.resource_review_store.root},
        passphrase="review-backup-passphrase",
    )
    assert receipt["cloud_backup_claimed"] is False
    restored = restore_encrypted_local_backup(
        payload,
        passphrase="review-backup-passphrase",
        destination=tmp_path / "restore-drill",
    )
    assert restored["all_hashes_verified"] is True
    assert (
        tmp_path
        / "restore-drill"
        / "restored"
        / "stores"
        / "resource_reviews"
        / f"{content_hash}.review.json"
    ).is_file()
