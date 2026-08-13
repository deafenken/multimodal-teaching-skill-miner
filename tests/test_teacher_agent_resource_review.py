from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_RECEIPT_SCHEMA,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_resource_review import (
    TeachingResourceReviewError,
    TeachingResourceReviewStore,
    build_reviewed_resource_projection,
    resource_descriptor_sha256,
    validate_reviewed_resource_projection,
)
from teaching_skill_miner.teacher_agent_resources import (
    extract_teaching_resource,
    teaching_resource_for_session,
    validate_teaching_resources,
)


NOW = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)


def _blocked_resource() -> dict:
    resource = extract_teaching_resource(
        b"slide body says 3; speaker notes say 4",
        "text/plain",
        display_name="conflict.txt",
    )
    text = (
        "[内容冲突待确认：正文写成 3 项，讲者备注写成 4 项]\n"
        "正文：系统包含三个部分。\n讲者备注：系统包含四个部分。"
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
                        "description": "正文写成 3 项，讲者备注写成 4 项",
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
    return resource


def _request(resource: dict, *, version: int = 0, key: str = "review-key-0001") -> dict:
    return {
        "resource_id": resource["resource_id"],
        "staged_resource_id": "stage_" + resource["content_sha256"][:24],
        "content_sha256": resource["content_sha256"],
        "expected_review_version": version,
        "resource_review_idempotency_key": key,
        "original_resource_sha256": resource_descriptor_sha256(resource),
        "reviewed_text": "教师对照原文件后确认：正文应为三个部分；讲者备注中的四是笔误。",
        "resolved_conflict_ids": ["resource_conflict_001"],
        "excluded_layer_ids": ["layer_002"],
        "attestations": {
            "compared_with_original_source": True,
            "uncertainties_removed_or_explicit": True,
            "not_an_answer_key": True,
            "context_only": True,
        },
        "review_note": "对照原始幻灯片核对正文与讲者备注。",
    }


def _receipt(request: dict, *, path: str = "api/resource/review") -> dict:
    material = {
        "schema": TEACHER_AUTHORITY_RECEIPT_SCHEMA,
        "authority_id": "tauth_" + "a" * 24,
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "assurance": AUTHORITY_ASSURANCE,
        "scope_id": "scope_" + "b" * 48,
        "scope_key_version": "k1",
        "actor_principal_sha256": "c" * 64,
        "roles_sha256": "d" * 64,
        "role_policy_sha256": "e" * 64,
        "method": "POST",
        "path": path,
        "body_sha256": canonical_sha256(request),
        "idempotency_key_sha256": "f" * 64,
        "issued_at": "2026-08-12T08:00:00Z",
        "expires_at": "2026-08-12T08:02:00Z",
        "nonce_sha256": "1" * 64,
        "gateway_envelope_sha256": "2" * 64,
        "service_authorization_signature_verified": True,
        "personal_non_repudiation": False,
        "signature": "A" * 43,
    }
    return {**material, "receipt_sha256": canonical_sha256(material)}


def test_authenticated_review_only_creates_untrusted_context_projection() -> None:
    original = _blocked_resource()
    request = _request(original)
    reviewed = build_reviewed_resource_projection(
        original,
        request,
        _receipt(request),
        review_version=1,
        previous_review_sha256=None,
        now=NOW,
    )

    assert (
        teaching_resource_for_session(reviewed)["resource_id"]
        == original["resource_id"]
    )
    assert reviewed["needs_review"] is False
    assert reviewed["requires_confirmation"] is False
    assert reviewed["evidence_contract"]["conflicts"] == []
    assert reviewed["evidence_contract"]["layers"][1]["status"] == "unavailable"
    receipt = reviewed["review_projection"]["receipt"]
    assert receipt["review_scope"] == "untrusted_teaching_context_only"
    assert receipt["semantic_understanding_established"] is False
    assert receipt["grading_evidence_allowed"] is False
    assert receipt["mastery_evidence_allowed"] is False
    assert receipt["personal_non_repudiation"] is False
    validate_reviewed_resource_projection(reviewed)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda request: request.update(resolved_conflict_ids=[]),
            "resolve every",
        ),
        (
            lambda request: request.update(reviewed_text="[视觉复核：请以后再看]"),
            "unresolved markers",
        ),
        (
            lambda request: request["attestations"].update(context_only=False),
            "attestations",
        ),
    ],
)
def test_review_fails_closed_without_actual_bounded_resolution(
    mutator, message
) -> None:
    original = _blocked_resource()
    request = _request(original)
    mutator(request)
    with pytest.raises(TeachingResourceReviewError, match=message):
        build_reviewed_resource_projection(
            original,
            request,
            _receipt(request),
            review_version=1,
            previous_review_sha256=None,
            now=NOW,
        )


def test_review_authority_is_bound_to_exact_path_and_body() -> None:
    original = _blocked_resource()
    request = _request(original)
    with pytest.raises(TeachingResourceReviewError, match="not bound"):
        build_reviewed_resource_projection(
            original,
            request,
            _receipt(request, path="api/adjudication/decide"),
            review_version=1,
            previous_review_sha256=None,
            now=NOW,
        )
    changed = deepcopy(request)
    changed["reviewed_text"] += "（被篡改）"
    with pytest.raises(TeachingResourceReviewError, match="not bound"):
        build_reviewed_resource_projection(
            original,
            changed,
            _receipt(request),
            review_version=1,
            previous_review_sha256=None,
            now=NOW,
        )


def test_review_store_is_restart_safe_idempotent_and_versioned(tmp_path: Path) -> None:
    original = _blocked_resource()
    first_request = _request(original)
    store = TeachingResourceReviewStore(tmp_path / "reviews")
    first = store.review(original, first_request, _receipt(first_request), now=NOW)

    replayed = TeachingResourceReviewStore(tmp_path / "reviews").review(
        original, first_request, _receipt(first_request), now=NOW
    )
    assert replayed == first
    assert (tmp_path / "reviews").stat().st_mode & 0o777 == 0o700
    assert (
        next((tmp_path / "reviews").glob("*.review.json")).stat().st_mode & 0o777
        == 0o600
    )

    stale = _request(original, version=0, key="review-key-0002")
    stale["reviewed_text"] = "第二次教师修订文本。"
    stale["original_resource_sha256"] = resource_descriptor_sha256(original)
    with pytest.raises(TeachingResourceReviewError, match="version conflict"):
        store.review(original, stale, _receipt(stale), now=NOW)

    second_request = {**stale, "expected_review_version": 1}
    second = store.review(
        original,
        second_request,
        _receipt(second_request),
        now=NOW,
    )
    assert second["review_projection"]["receipt"]["review_version"] == 2
    assert store.get(original["content_sha256"]) == second
    assert store.get_by_resource_id(original["resource_id"]) == second


def test_review_store_tampering_and_symlink_roots_fail_closed(tmp_path: Path) -> None:
    original = _blocked_resource()
    request = _request(original)
    root = tmp_path / "reviews"
    store = TeachingResourceReviewStore(root)
    store.review(original, request, _receipt(request), now=NOW)
    document = next(root.glob("*.review.json"))
    document.write_text(document.read_text().replace("context_only", "context_only_x"))
    with pytest.raises(TeachingResourceReviewError, match="integrity"):
        TeachingResourceReviewStore(root).get(original["content_sha256"])

    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "review-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(TeachingResourceReviewError, match="symlink"):
        TeachingResourceReviewStore(link)
