from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_consent import (
    ConsentError,
    RemoteConsentStore,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value


def _store(tmp_path, clock: _Clock | None = None) -> RemoteConsentStore:
    return RemoteConsentStore(
        tmp_path / "private" / "consent.json",
        signing_secret=b"c" * 32,
        clock=clock or _Clock(),
    )


def test_server_receipt_is_purpose_provider_subject_and_data_bound(tmp_path) -> None:
    store = _store(tmp_path)
    receipt = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_visual_analysis",
        provider_id="local-vision-v1",
        processing_region="on_device",
        data_categories=["learner_image"],
        provider_retention_days=0,
    )

    verified = store.verify(
        receipt["consent_id"],
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_visual_analysis",
        provider_id="local-vision-v1",
        required_data_categories=["learner_image"],
    )
    assert verified == receipt
    assert receipt["signature"]
    assert receipt["provider_policy_sha256"]
    assert receipt["subject_policy_sha256"]
    assert (
        receipt["provider_policy"]["policy_source"]
        == "local_unverified_test_or_standalone"
    )
    with pytest.raises(ConsentError, match="authorize"):
        store.verify(
            receipt["consent_id"],
            subject_id="subject_otherabcdefghij",
            purpose="remote_visual_analysis",
            provider_id="local-vision-v1",
            required_data_categories=["learner_image"],
        )
    with pytest.raises(ConsentError, match="authorize"):
        store.verify(
            receipt["consent_id"],
            subject_id="subject_abcdefghijklmnop",
            purpose="public_web_search",
            provider_id="local-vision-v1",
            required_data_categories=["learner_image"],
        )


def test_revocation_and_expiry_fail_closed_across_restart(tmp_path) -> None:
    clock = _Clock()
    store = _store(tmp_path, clock)
    first = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_chat",
        provider_id="deepseek-v4-flash",
        processing_region="provider_managed",
        data_categories=["learner_message"],
        provider_retention_days=30,
        validity_days=2,
    )
    revoked = store.revoke(
        first["consent_id"],
        subject_id="subject_abcdefghijklmnop",
        reason_code="user_withdrew",
    )
    assert revoked["status"] == "revoked"
    restarted = _store(tmp_path, clock)
    with pytest.raises(ConsentError, match="revoked"):
        restarted.verify(
            first["consent_id"],
            subject_id="subject_abcdefghijklmnop",
            purpose="remote_chat",
            provider_id="deepseek-v4-flash",
            required_data_categories=["learner_message"],
        )

    second = restarted.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_teaching",
        provider_id="deepseek-v4-flash",
        processing_region="provider_managed",
        data_categories=["learner_message", "learner_profile_bounded"],
        provider_retention_days=30,
        validity_days=1,
    )
    clock.value += timedelta(days=2)
    with pytest.raises(ConsentError, match="expired"):
        restarted.verify(
            second["consent_id"],
            subject_id="subject_abcdefghijklmnop",
            purpose="remote_teaching",
            provider_id="deepseek-v4-flash",
            required_data_categories=["learner_message"],
        )


def test_likely_minor_requires_guardian_or_school_policy(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ConsentError, match="guardian or school"):
        store.grant(
            subject_id="subject_minorabcdefghijk",
            purpose="remote_teaching",
            provider_id="deepseek-v4-flash",
            processing_region="provider_managed",
            data_categories=["learner_message"],
            provider_retention_days=30,
            likely_minor=True,
        )
    receipt = store.grant(
        subject_id="subject_minorabcdefghijk",
        purpose="remote_teaching",
        provider_id="deepseek-v4-flash",
        processing_region="provider_managed",
        data_categories=["learner_message"],
        provider_retention_days=30,
        likely_minor=True,
        guardian_or_school_policy="verified_school_policy",
    )
    assert receipt["guardian_or_school_policy"] == "verified_school_policy"


def test_deployment_provider_and_subject_policy_are_hash_bound(tmp_path) -> None:
    store = _store(tmp_path)
    receipt = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_chat",
        provider_id="deepseek",
        processing_region="cn_north",
        data_categories=["learner_message"],
        provider_retention_days=7,
        provider_policy={
            "policy_id": "deepseek-education-policy",
            "policy_version": "2026-08-12",
            "policy_source": "deployment_operator_asserted_external_terms_not_repository_verified",
            "processing_region": "cn_north",
            "provider_retention_days": 7,
            "deletion_status": "outside_service_control_subject_to_provider_policy",
            "documentation_url": "https://provider.example/privacy",
        },
        subject_policy={
            "policy_id": "school-roster-policy",
            "policy_version": "2026-fall",
            "policy_source": "organization_oidc_or_roster_policy",
            "likely_minor": True,
            "guardian_or_school_policy": "verified_school_policy",
            "remote_processing_eligible": True,
        },
        likely_minor=True,
        guardian_or_school_policy="verified_school_policy",
    )
    assert receipt["provider_policy"]["policy_id"] == "deepseek-education-policy"
    assert receipt["subject_policy"]["likely_minor"] is True
    assert (
        store.verify(
            receipt["consent_id"],
            subject_id="subject_abcdefghijklmnop",
            purpose="remote_chat",
            provider_id="deepseek",
            required_data_categories=["learner_message"],
            provider_policy=receipt["provider_policy"],
            subject_policy=receipt["subject_policy"],
        )
        == receipt
    )
    changed = dict(receipt["provider_policy"])
    changed["policy_version"] = "2026-08-13"
    with pytest.raises(ConsentError, match="stale"):
        store.verify(
            receipt["consent_id"],
            subject_id="subject_abcdefghijklmnop",
            purpose="remote_chat",
            provider_id="deepseek",
            required_data_categories=["learner_message"],
            provider_policy=changed,
            subject_policy=receipt["subject_policy"],
        )


def test_store_contains_no_learner_content_and_tampering_is_rejected(tmp_path) -> None:
    store = _store(tmp_path)
    store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="public_web_search",
        provider_id="anthropic-web-search",
        processing_region="provider_managed",
        data_categories=["public_web_query"],
        provider_retention_days=0,
    )
    path = tmp_path / "private" / "consent.json"
    serialized = path.read_text(encoding="utf-8")
    assert "我的隐私学习内容" not in serialized
    payload = json.loads(serialized)
    payload["events"][0]["receipt"]["provider_id"] = "attacker"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ConsentError, match="seal|chain"):
        store.list_for_subject("subject_abcdefghijklmnop")


def test_public_consent_receipt_schema_validates_server_receipt(tmp_path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    store = _store(tmp_path)
    receipt = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_chat",
        provider_id="deepseek-v4-flash",
        processing_region="provider_managed",
        data_categories=["learner_message"],
        provider_retention_days=30,
    )
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schema/teacher_agent_remote_consent_receipt.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(receipt)
