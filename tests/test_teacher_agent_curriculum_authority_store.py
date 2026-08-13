from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
from pathlib import Path
import http.client
import io
import threading
import time
from urllib.parse import urlsplit
import zipfile

import pytest
from jsonschema import Draft202012Validator

from teaching_skill_miner.student_model import initialize_student_model
from teaching_skill_miner.teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_SCHEMA,
    TeacherAuthorityVerifier,
    canonical_bytes,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent import _refresh_integrity, validate_session
from teaching_skill_miner.teacher_agent_curriculum import (
    build_teacher_owned_curriculum_spec,
    create_teacher_curriculum_authority_receipt,
    seal_teacher_owned_curriculum_blueprint,
)
from teaching_skill_miner.teacher_agent_curriculum_authority_store import (
    CurriculumAuthorityConflictError,
    CurriculumAuthorityStoreError,
    TeachingCurriculumAuthorityStore,
    curriculum_runtime_authority_projection,
)
from teaching_skill_miner.teacher_agent_curriculum_signing import (
    CurriculumSigningKeyring,
    CurriculumSigningKeyringError,
)
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)
from teaching_skill_miner.teacher_agent_data_rights import (
    deletion_confirmation,
    validate_project_export_archive,
)
from teaching_skill_miner.teacher_agent_syllabus import _seal_generated_syllabus
from teaching_skill_miner.teacher_agent_syllabus import teaching_syllabus_editable_draft


NOW = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc)
SCOPE = "scope_" + "a" * 48
OTHER_SCOPE = "scope_" + "b" * 48
AUTHORITY_KEY = b"g" * 32
KEYRING_KEY = b"k" * 32
BINDING = {
    "family_id": "syf_" + "1" * 24,
    "published_revision_id": "syr_" + "2" * 24,
    "published_syllabus_id": "syl_" + "3" * 24,
    "published_syllabus_sha256": "4" * 64,
}


def _source() -> dict:
    return {
        "resource_id": "teacher-source",
        "content_sha256": sha256(b"document").hexdigest(),
        "excerpt_sha256": sha256(b"excerpt").hexdigest(),
        "locator": {"kind": "page", "start": 2, "end": 2},
    }


def _spec(*, label: str = "反应输入") -> dict:
    return build_teacher_owned_curriculum_spec(
        title="光合作用",
        lessons=[
            {
                "legacy_lesson_id": "lesson_01_01",
                "title": "反应输入课",
                "objective": "识别光合作用的反应输入。",
                "knowledge_components": [
                    {
                        "label": label,
                        "prerequisites": [],
                        "source_resource_ids": ["teacher-source"],
                    }
                ],
            }
        ],
        source_spans=[_source()],
        factual_claims=[
            {
                "statement": "植物利用光能把二氧化碳和水转化为有机物。",
                "knowledge_components": [label],
                "source_resource_ids": ["teacher-source"],
            }
        ],
    )


def _verifier(root: Path, *, scope: str = SCOPE) -> TeacherAuthorityVerifier:
    return TeacherAuthorityVerifier(
        key=AUTHORITY_KEY,
        scope_id=scope,
        scope_key_version="k1",
        replay_store_path=root / f"authority-{scope[-1]}.jsonl",
        clock=lambda: NOW,
    )


def _gateway_receipt(
    verifier: TeacherAuthorityVerifier,
    *,
    path: str,
    body: dict,
    nonce: str,
) -> dict:
    idempotency_key = body["curriculum_authority_idempotency_key"]
    envelope = {
        "schema": TEACHER_AUTHORITY_SCHEMA,
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "assurance": AUTHORITY_ASSURANCE,
        "scope_id": verifier.scope_id,
        "scope_key_version": verifier.scope_key_version,
        "actor_principal_sha256": "5" * 64,
        "roles_sha256": "6" * 64,
        "role_policy_sha256": "7" * 64,
        "method": "POST",
        "path": path,
        "body_sha256": canonical_sha256(body),
        "idempotency_key_sha256": sha256(idempotency_key.encode()).hexdigest(),
        "issued_at": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "expires_at": (NOW + timedelta(minutes=2))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "nonce": "tan_" + nonce * 48,
    }
    envelope["authority_id"] = "tauth_" + canonical_sha256(envelope)[:24]
    envelope["signature"] = (
        base64.urlsafe_b64encode(
            hmac.new(AUTHORITY_KEY, canonical_bytes(envelope), sha256).digest()
        )
        .decode()
        .rstrip("=")
    )
    return verifier.verify(
        {**deepcopy(body), "_teacher_authority": envelope},
        method="POST",
        path=path,
    )


def _gateway_request(
    verifier: TeacherAuthorityVerifier,
    *,
    path: str,
    body: dict,
    nonce: str,
) -> dict:
    idempotency_key = body["curriculum_authority_idempotency_key"]
    envelope = {
        "schema": TEACHER_AUTHORITY_SCHEMA,
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "assurance": AUTHORITY_ASSURANCE,
        "scope_id": verifier.scope_id,
        "scope_key_version": verifier.scope_key_version,
        "actor_principal_sha256": "5" * 64,
        "roles_sha256": "6" * 64,
        "role_policy_sha256": "7" * 64,
        "method": "POST",
        "path": path,
        "body_sha256": canonical_sha256(body),
        "idempotency_key_sha256": sha256(idempotency_key.encode()).hexdigest(),
        "issued_at": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "expires_at": (NOW + timedelta(minutes=2))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "nonce": "tan_" + nonce * 48,
    }
    envelope["authority_id"] = "tauth_" + canonical_sha256(envelope)[:24]
    envelope["signature"] = (
        base64.urlsafe_b64encode(
            hmac.new(AUTHORITY_KEY, canonical_bytes(envelope), sha256).digest()
        )
        .decode()
        .rstrip("=")
    )
    return {**deepcopy(body), "_teacher_authority": envelope}


def _store(root: Path, keyring: CurriculumSigningKeyring) -> TeachingCurriculumAuthorityStore:
    verifier = _verifier(root)
    return TeachingCurriculumAuthorityStore(
        root / "curriculum-authority.json",
        scope_id=SCOPE,
        trusted_teacher_public_keys=keyring.trusted_public_keys,
        known_teacher_public_keys=keyring.all_public_keys,
        gateway_receipt_validator=verifier.verify_verification_receipt,
    )


def _review(
    root: Path,
    store: TeachingCurriculumAuthorityStore,
    spec: dict,
    *,
    key: str = "curriculum-review-0001",
    expected: int = 0,
    nonce: str = "a",
) -> tuple[dict, bool]:
    body = {
        "syllabus_id": BINDING["published_syllabus_id"],
        "teacher_spec": spec,
        "expected_syllabus_version": 1,
        "expected_authority_version": expected,
        "curriculum_authority_idempotency_key": key,
    }
    receipt = _gateway_receipt(
        _verifier(root), path="api/curriculum/review", body=body, nonce=nonce
    )
    return store.record_review(
        spec,
        **BINDING,
        expected_version=expected,
        idempotency_key=key,
        gateway_authority_receipt=receipt,
    )


def _seal(
    root: Path,
    store: TeachingCurriculumAuthorityStore,
    keyring: CurriculumSigningKeyring,
    review: dict,
    *,
    key: str = "curriculum-seal-0001",
    nonce: str = "b",
) -> tuple[dict, bool]:
    signing_key_id, private_pem = keyring.active_signing_material()
    receipt = create_teacher_curriculum_authority_receipt(
        review["review"]["teacher_spec"],
        teacher_id_hash="5" * 64,
        reviewed_at="2026-08-12T04:00:00Z",
        teacher_confirmed_authority=True,
        signing_key_id=signing_key_id,
        private_key_pem=private_pem,
    )
    blueprint = seal_teacher_owned_curriculum_blueprint(
        review["review"]["teacher_spec"],
        receipt,
        trusted_teacher_public_keys=keyring.trusted_public_keys(),
    )
    body = {
        "syllabus_id": BINDING["published_syllabus_id"],
        "review_id": review["review"]["review_id"],
        "teacher_confirmed_authority": True,
        "expected_syllabus_version": 1,
        "expected_authority_version": review["version"],
        "curriculum_authority_idempotency_key": key,
    }
    gateway = _gateway_receipt(
        _verifier(root), path="api/curriculum/seal", body=body, nonce=nonce
    )
    return store.seal_review(
        blueprint,
        **BINDING,
        review_id=review["review"]["review_id"],
        teacher_spec_sha256=review["review"]["teacher_spec_sha256"],
        expected_version=review["version"],
        idempotency_key=key,
        gateway_authority_receipt=gateway,
    )


def test_review_seal_restart_projection_cas_and_exact_idempotency(tmp_path: Path) -> None:
    keyring = CurriculumSigningKeyring(tmp_path / "keyring.json", integrity_key=KEYRING_KEY)
    store = _store(tmp_path, keyring)
    spec = _spec()
    review, created = _review(tmp_path, store, spec)
    assert created and review["status"] == "reviewed" and review["version"] == 1

    # Exact retry recovers the committed projection even with a fresh gateway receipt.
    replay, created = _review(
        tmp_path, store, spec, nonce="c"
    )
    assert not created and replay == review
    with pytest.raises(CurriculumAuthorityConflictError, match="reused"):
        _review(
            tmp_path,
            store,
            _spec(label="另一知识组件"),
            nonce="d",
        )

    sealed, created = _seal(tmp_path, store, keyring, review)
    assert created and sealed["status"] == "sealed" and sealed["version"] == 2
    restarted_keyring = CurriculumSigningKeyring(
        tmp_path / "keyring.json", integrity_key=KEYRING_KEY
    )
    restarted = _store(tmp_path, restarted_keyring)
    blueprint = restarted.active_blueprint(
        **BINDING, expected_authority_version=2
    )
    assert blueprint is not None
    projection = curriculum_runtime_authority_projection(
        blueprint,
        legacy_lesson_id="lesson_01_01",
        **BINDING,
        authority_version=2,
        trusted_teacher_public_keys=restarted_keyring.trusted_public_keys(),
    )
    assert projection["authority"] is True
    assert projection["factual_claim_ids"]
    assert projection["projection_sha256"] == canonical_sha256(
        {key: value for key, value in projection.items() if key != "projection_sha256"}
    )

    old_key = str(blueprint["authority"]["receipt"]["signing_key_id"])
    new_key = restarted_keyring.rotate()
    assert new_key != old_key
    # Rotation keeps existing receipts trusted until an explicit old-key
    # revocation, while new seals immediately use the active key.
    assert restarted.active_blueprint(**BINDING, expected_authority_version=2)
    restarted_keyring.revoke(old_key)
    with pytest.raises(
        CurriculumAuthorityStoreError, match="verification failed|invalid"
    ):
        restarted.active_blueprint(**BINDING, expected_authority_version=2)
    assert old_key not in restarted_keyring.trusted_public_keys()
    assert old_key in restarted_keyring.all_public_keys()
    assert all(
        "public_key_base64" not in row and "private_key" not in row
        for row in restarted_keyring.public_status()["keys"]
    )

    # Revoking a signing key invalidates seals signed by that key without
    # bricking the historic ledger. The same scope can revoke that curriculum,
    # review a replacement, seal with the new active key, and restart cleanly.
    revoke_body = {
        "syllabus_id": BINDING["published_syllabus_id"],
        "curriculum_id": sealed["seal"]["curriculum_id"],
        "reason_code": "signing_key_revoked",
        "expected_syllabus_version": 1,
        "expected_authority_version": 2,
        "curriculum_authority_idempotency_key": "rotate-revoke-curriculum-0001",
    }
    revoke_receipt = _gateway_receipt(
        _verifier(tmp_path),
        path="api/curriculum/revoke",
        body=revoke_body,
        nonce="e",
    )
    revoked, created = restarted.revoke(
        **BINDING,
        curriculum_id=sealed["seal"]["curriculum_id"],
        reason_code="signing_key_revoked",
        expected_version=2,
        idempotency_key="rotate-revoke-curriculum-0001",
        gateway_authority_receipt=revoke_receipt,
    )
    assert created and revoked["status"] == "revoked" and revoked["version"] == 3
    replacement_review, _ = _review(
        tmp_path,
        restarted,
        _spec(label="新知识组件"),
        key="rotate-review-replacement-0001",
        expected=3,
        nonce="f",
    )
    replacement_seal, _ = _seal(
        tmp_path,
        restarted,
        restarted_keyring,
        replacement_review,
        key="rotate-seal-replacement-0001",
        nonce="9",
    )
    assert (
        replacement_seal["seal"]["curriculum_blueprint"]["authority"]["receipt"][
            "signing_key_id"
        ]
        == new_key
    )
    post_rotation_keyring = CurriculumSigningKeyring(
        tmp_path / "keyring.json", integrity_key=KEYRING_KEY
    )
    post_rotation_store = _store(tmp_path, post_rotation_keyring)
    assert post_rotation_store.active_blueprint(
        **BINDING, expected_authority_version=5
    ) is not None


def test_revoke_is_durable_and_active_blueprint_fails_closed(tmp_path: Path) -> None:
    keyring = CurriculumSigningKeyring(tmp_path / "keyring.json", integrity_key=KEYRING_KEY)
    store = _store(tmp_path, keyring)
    review, _ = _review(tmp_path, store, _spec())
    sealed, _ = _seal(tmp_path, store, keyring, review)
    body = {
        "syllabus_id": BINDING["published_syllabus_id"],
        "curriculum_id": sealed["seal"]["curriculum_id"],
        "reason_code": "teacher_revoked",
        "expected_syllabus_version": 1,
        "expected_authority_version": 2,
        "curriculum_authority_idempotency_key": "curriculum-revoke-0001",
    }
    receipt = _gateway_receipt(
        _verifier(tmp_path), path="api/curriculum/revoke", body=body, nonce="e"
    )
    revoked, created = store.revoke(
        **BINDING,
        curriculum_id=sealed["seal"]["curriculum_id"],
        reason_code="teacher_revoked",
        expected_version=2,
        idempotency_key="curriculum-revoke-0001",
        gateway_authority_receipt=receipt,
    )
    assert created and revoked["status"] == "revoked" and revoked["version"] == 3
    with pytest.raises(CurriculumAuthorityStoreError, match="not actively sealed"):
        _store(tmp_path, keyring).active_blueprint(**BINDING)


def test_store_tamper_cross_scope_and_truncation_fail_closed(tmp_path: Path) -> None:
    keyring = CurriculumSigningKeyring(tmp_path / "keyring.json", integrity_key=KEYRING_KEY)
    store = _store(tmp_path, keyring)
    _review(tmp_path, store, _spec())
    copied = tmp_path / "copied.json"
    copied.write_bytes((tmp_path / "curriculum-authority.json").read_bytes())
    other = TeachingCurriculumAuthorityStore(
        copied,
        scope_id=OTHER_SCOPE,
        trusted_teacher_public_keys=keyring.trusted_public_keys,
        gateway_receipt_validator=_verifier(tmp_path).verify_verification_receipt,
    )
    with pytest.raises(CurriculumAuthorityStoreError, match="boundary"):
        other.read_family(BINDING["family_id"])

    original = json.loads((tmp_path / "curriculum-authority.json").read_text())
    original["events"][0]["payload"]["teacher_spec_sha256"] = "0" * 64
    (tmp_path / "curriculum-authority.json").write_text(json.dumps(original))
    with pytest.raises(CurriculumAuthorityStoreError):
        store.read_family(BINDING["family_id"])
    (tmp_path / "curriculum-authority.json").write_bytes(b'{"schema":')
    with pytest.raises(CurriculumAuthorityStoreError, match="cannot be read"):
        store.read_family(BINDING["family_id"])

    unsafe = tmp_path / "unsafe-ledger.json"
    unsafe.symlink_to(tmp_path / "curriculum-authority.json")
    with pytest.raises(CurriculumAuthorityStoreError, match="symlink"):
        TeachingCurriculumAuthorityStore(
            unsafe,
            scope_id=SCOPE,
            trusted_teacher_public_keys=keyring.trusted_public_keys,
            gateway_receipt_validator=_verifier(tmp_path).verify_verification_receipt,
        )


def test_keyring_aead_rotation_revocation_restart_and_trust_lease(tmp_path: Path) -> None:
    path = tmp_path / "keyring.json"
    keyring = CurriculumSigningKeyring(path, integrity_key=KEYRING_KEY)
    first, first_pem = keyring.active_signing_material()
    disk = path.read_text()
    assert "PRIVATE KEY" not in disk and "private_key_pem" not in disk
    assert "private_key_nonce_base64" in disk
    second = keyring.rotate()
    assert first != second and set(keyring.trusted_public_keys()) == {first, second}

    entered = threading.Event()
    release = threading.Event()
    revoked = threading.Event()

    def commit_reader() -> None:
        with keyring.trusted_public_keys_lease() as keys:
            assert first in keys
            entered.set()
            assert release.wait(2)

    def revoke_writer() -> None:
        assert entered.wait(2)
        keyring.revoke(first)
        revoked.set()

    reader = threading.Thread(target=commit_reader)
    writer = threading.Thread(target=revoke_writer)
    reader.start()
    writer.start()
    assert entered.wait(2)
    time.sleep(0.05)
    assert not revoked.is_set()
    release.set()
    reader.join(2)
    writer.join(2)
    assert revoked.is_set() and set(keyring.trusted_public_keys()) == {second}
    with pytest.raises(CurriculumSigningKeyringError, match="rotate"):
        keyring.revoke(second)
    restarted = CurriculumSigningKeyring(path, integrity_key=KEYRING_KEY)
    assert restarted.active_signing_material()[0] == second
    assert first_pem not in path.read_bytes()

    tampered = json.loads(path.read_text())
    active = tampered["active_key_id"]
    ciphertext = tampered["keys"][active]["private_key_ciphertext_base64"]
    tampered["keys"][active]["private_key_ciphertext_base64"] = (
        ("A" if ciphertext[0] != "A" else "B") + ciphertext[1:]
    )
    path.write_text(json.dumps(tampered))
    with pytest.raises(CurriculumSigningKeyringError):
        CurriculumSigningKeyring(path, integrity_key=KEYRING_KEY)

    # Even a party capable of recomputing the outer integrity tag cannot swap
    # the nonce/key identity/public-key AAD without failing AES-GCM.
    isolated = tmp_path / "isolated-keyring.json"
    guarded = CurriculumSigningKeyring(isolated, integrity_key=KEYRING_KEY)
    guarded.rotate()
    swapped = json.loads(isolated.read_text())
    left, right = list(swapped["keys"])
    swapped["keys"][left]["private_key_nonce_base64"] = swapped["keys"][right][
        "private_key_nonce_base64"
    ]
    material = deepcopy(swapped)
    material.pop("integrity_hmac_sha256")
    swapped["integrity_hmac_sha256"] = hmac.new(
        guarded._integrity_key,
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(),
        sha256,
    ).hexdigest()
    isolated.write_text(json.dumps(swapped))
    with pytest.raises(CurriculumSigningKeyringError, match="envelope"):
        CurriculumSigningKeyring(isolated, integrity_key=KEYRING_KEY)


def test_authority_store_public_schema_and_revoke_commit_lease(tmp_path: Path) -> None:
    keyring = CurriculumSigningKeyring(tmp_path / "keyring.json", integrity_key=KEYRING_KEY)
    store = _store(tmp_path, keyring)
    review, _ = _review(tmp_path, store, _spec())
    sealed, _ = _seal(tmp_path, store, keyring, review)
    schema = json.loads(
        (project_root() / "schema/teacher_agent_curriculum_authority_store.schema.json")
        .read_text()
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(
        json.loads((tmp_path / "curriculum-authority.json").read_text())
    )
    assert "private_key" not in json.dumps(schema)

    body = {
        "syllabus_id": BINDING["published_syllabus_id"],
        "curriculum_id": sealed["seal"]["curriculum_id"],
        "reason_code": "teacher_revoked",
        "expected_syllabus_version": 1,
        "expected_authority_version": 2,
        "curriculum_authority_idempotency_key": "lease-revoke-0001",
    }
    receipt = _gateway_receipt(
        _verifier(tmp_path), path="api/curriculum/revoke", body=body, nonce="9"
    )
    entered = threading.Event()
    release = threading.Event()
    revoked = threading.Event()

    def commit_reader() -> None:
        with store.active_blueprint_lease(**BINDING, expected_authority_version=2):
            entered.set()
            assert release.wait(2)

    def revoke_writer() -> None:
        assert entered.wait(2)
        store.revoke(
            **BINDING,
            curriculum_id=sealed["seal"]["curriculum_id"],
            reason_code="teacher_revoked",
            expected_version=2,
            idempotency_key="lease-revoke-0001",
            gateway_authority_receipt=receipt,
        )
        revoked.set()

    reader = threading.Thread(target=commit_reader)
    writer = threading.Thread(target=revoke_writer)
    reader.start()
    writer.start()
    assert entered.wait(2)
    time.sleep(0.05)
    assert not revoked.is_set()
    release.set()
    reader.join(2)
    writer.join(2)
    assert revoked.is_set()


def _http_post(base_url: str, route: str, body: dict) -> tuple[int, dict]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    encoded = json.dumps(body, ensure_ascii=False).encode()
    connection.request(
        "POST",
        parsed.path + route,
        body=encoded,
        headers={"Content-Type": "application/json", "Content-Length": str(len(encoded))},
    )
    response = connection.getresponse()
    payload = json.loads(response.read())
    connection.close()
    return response.status, payload


def test_authenticated_dashboard_review_seal_start_and_browser_forgery_boundary(
    tmp_path: Path,
) -> None:
    keyring = CurriculumSigningKeyring(
        tmp_path / "syllabi" / ".curriculum_signing_keyring.json",
        integrity_key=KEYRING_KEY,
    )
    verifier = _verifier(tmp_path)
    authority_store = TeachingCurriculumAuthorityStore(
        tmp_path / "syllabi" / ".curriculum_authority.json",
        scope_id=SCOPE,
        trusted_teacher_public_keys=keyring.trusted_public_keys,
        gateway_receipt_validator=verifier.verify_verification_receipt,
    )
    data = project_root() / "data"
    snapshot = build_teacher_agent_dashboard_snapshot(
        data / "teacher_agent_skill_library_v2.json",
        data / "teacher_agent_demo_input.json",
        data / "teacher_agent_evaluation_cases.json",
        store_path=tmp_path / "sessions.jsonl",
        syllabus_store_path=tmp_path / "syllabi",
        project_store_path=tmp_path / "projects",
        learning_record_store_path=tmp_path / "learning.jsonl",
        metacognition_store_path=tmp_path / "metacognition.jsonl",
        learner_key_secret=b"curriculum-authority-learner-key-secret",
        learner_tenant_id="school-curriculum-authority",
        trusted_learner_profile_ref="profile_" + "f" * 64,
        teacher_authority_verifier=verifier,
        curriculum_authority_store=authority_store,
        curriculum_signing_keyring=keyring,
    )
    syllabus = _seal_generated_syllabus(
        {
            "title": "光合作用",
            "description": "识别光合作用输入。",
            "audience": "初学者",
            "estimated_duration_minutes": 30,
            "learning_objectives": ["识别光合作用的反应输入"],
            "prerequisites": [],
            "modules": [
                {
                    "title": "输入",
                    "description": "反应输入",
                    "lessons": [
                        {
                            "title": "反应输入课",
                            "objective": "识别光合作用的反应输入。",
                            "summary": "识别必要输入。",
                            "duration_minutes": 30,
                            "knowledge_components": ["反应输入"],
                            "materials": {
                                "example": "观察叶片。",
                                "practice": "列出输入。",
                                "transfer_task": "解释新情境。",
                            },
                        }
                    ],
                }
            ],
        },
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-12T01:00:00Z",
    )
    imported = snapshot.import_syllabus({"syllabus": syllabus})
    family = imported["version_family"]
    spec = _spec()
    review_body = {
        "syllabus_id": syllabus["syllabus_id"],
        "teacher_spec": spec,
        "expected_syllabus_version": family["version"],
        "expected_authority_version": 0,
        "curriculum_authority_idempotency_key": "dashboard-review-0001",
    }
    reviewed = snapshot.review_curriculum(
        _gateway_request(
            verifier,
            path="api/curriculum/review",
            body=review_body,
            nonce="f",
        )
    )
    seal_body = {
        "syllabus_id": syllabus["syllabus_id"],
        "review_id": reviewed["curriculum_authority"]["review"]["review_id"],
        "teacher_confirmed_authority": True,
        "expected_syllabus_version": family["version"],
        "expected_authority_version": 1,
        "curriculum_authority_idempotency_key": "dashboard-seal-0001",
    }
    sealed = snapshot.seal_curriculum(
        _gateway_request(
            verifier,
            path="api/curriculum/seal",
            body=seal_body,
            nonce="0",
        )
    )
    assert sealed["authoritative_for_runtime_grading"] is True
    payload = snapshot.syllabus_lesson_payload(
        syllabus["syllabus_id"], "lesson_01_01"
    )
    authority = payload["goal"]["curriculum_authority"]
    assert authority["authority_version"] == 2
    assert payload["goal"]["knowledge_spec"]["rubric_criteria"]
    started = snapshot.start(
        {
            "goal": payload["goal"],
            "syllabus_ref": payload["syllabus_ref"],
            "student_profile": {"learner_level": "beginner"},
            "start_idempotency_key": "sealed-dashboard-start-0001",
        }
    )
    assert (
        started["setup_snapshot"]["goal"]["knowledge_spec"]["status"]
        == "sealed_teacher_curriculum"
    )

    project = snapshot.create_project({"title": "密封课程审计项目"})["project"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "teaching_session", "reference_id": started["session_id"]},
    )
    archive = snapshot.export_project(project["project_id"])
    validate_project_export_archive(archive.payload)
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        authority_path = f"curriculum_authorities/{family['family_id']}.json"
        assert authority_path in exported.namelist()
        exported_authority = exported.read(authority_path)
        assert b"curriculum_authority_projection.v1" in exported_authority
        assert b"private_key" not in exported_authority
        assert b"private_key_ciphertext_base64" not in archive.payload

    forged = deepcopy(snapshot.bootstrap()["default_goal"])
    forged["knowledge_spec"] = deepcopy(spec)
    with pytest.raises(TeacherAgentDashboardError, match="cannot supply grading authority"):
        snapshot.start(
            {
                "goal": forged,
                "student_profile": {"learner_level": "beginner"},
                "start_idempotency_key": "forged-dashboard-start-0001",
            }
        )
    assert len(snapshot.sessions) == 1

    server, base_url = create_teacher_agent_dashboard_server(snapshot)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, rejected = _http_post(
            base_url,
            "api/start",
            {
                "goal": forged,
                "student_profile": {"learner_level": "beginner"},
                "start_idempotency_key": "forged-http-start-0001",
            },
        )
        assert status == 400
        assert "cannot supply grading authority" in rejected["error"]
        assert len(snapshot.sessions) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    first_draft = teaching_syllabus_editable_draft(syllabus)
    first_draft["description"] = "修订后的光合作用输入说明。"
    revision_two = snapshot.revise_syllabus(
        syllabus["syllabus_id"],
        {
            "editable_draft": first_draft,
            "change_summary": "建立第二版",
            "expected_version": 1,
            "idempotency_key": "curriculum-stale-revision-0001",
        },
    )
    second_draft = teaching_syllabus_editable_draft(syllabus)
    second_draft["description"] = "第三版光合作用输入说明。"
    revision_three = snapshot.revise_syllabus(
        syllabus["syllabus_id"],
        {
            "editable_draft": second_draft,
            "change_summary": "建立第三版",
            "expected_version": 2,
            "idempotency_key": "curriculum-stale-revision-0002",
        },
    )
    revision_two_id = next(
        row["revision_id"]
        for row in revision_two["version_family"]["revisions"]
        if row["syllabus_id"] == revision_two["syllabus"]["syllabus_id"]
    )
    revision_three_id = next(
        row["revision_id"]
        for row in revision_three["version_family"]["revisions"]
        if row["syllabus_id"] == revision_three["syllabus"]["syllabus_id"]
    )
    snapshot.publish_syllabus(
        syllabus["syllabus_id"],
        {
            "revision_id": revision_two_id,
            "expected_version": 3,
            "idempotency_key": "curriculum-stale-publish-0001",
        },
    )
    with pytest.raises(TeacherAgentDashboardError, match="currently published"):
        snapshot.syllabus_lesson_payload(syllabus["syllabus_id"], "lesson_01_01")
    snapshot.rollback_syllabus(
        revision_two["syllabus"]["syllabus_id"],
        {
            "revision_id": revision_three_id,
            "expected_version": 4,
            "idempotency_key": "curriculum-stale-rollback-0001",
        },
    )
    with pytest.raises(TeacherAgentDashboardError, match="currently published"):
        snapshot.syllabus_lesson_payload(syllabus["syllabus_id"], "lesson_01_01")
    initial_revision_id = next(
        row["revision_id"]
        for row in family["revisions"]
        if row["syllabus_id"] == syllabus["syllabus_id"]
    )
    snapshot.rollback_syllabus(
        revision_three["syllabus"]["syllabus_id"],
        {
            "revision_id": initial_revision_id,
            "expected_version": 5,
            "idempotency_key": "curriculum-restore-rollback-0001",
        },
    )
    assert snapshot.syllabus_lesson_payload(
        syllabus["syllabus_id"], "lesson_01_01"
    )["goal"]["curriculum_authority"]["authority_version"] == 2

    # A pre-answer judgment of learning may be recorded only while the exact
    # sealed curriculum is active.  Revocation after that JOL must also block
    # the later outcome-pairing boundary, even though the prediction itself is
    # already durable.
    record = snapshot.sessions[started["session_id"]]
    with record.lock:
        assessment_skill = next(
            item
            for item in snapshot.library["skills"]
            if item["skill_id"] == "skill_socratic_understanding_check"
        )
        action = record.session["current_action"]
        action["primary_skill"] = deepcopy(assessment_skill)
        action["knowledge_components"] = ["反应输入"]
        action["teacher_action"]["question_contract"] = {
            "answer_type": "explanation",
            "target_concepts": ["反应输入"],
            "accepted_aliases": [],
            "success_criteria": ["说明光能、二氧化碳和水是必要输入"],
            "grading_scope": "current_question_only",
        }
        record.session["student_state"]["student_model"] = initialize_student_model(
            record.session["student_state"].get("knowledge_mastery", {}),
            goal=record.session["goal"],
        )
        record.session = _refresh_integrity(record.session)
        validate_session(record.session)
        metacognition_guards = snapshot._response(started["session_id"], record)
    prediction = snapshot.record_metacognitive_prediction(
        {
            "session_id": started["session_id"],
            "expected_round": metacognition_guards["rounds_completed"],
            "expected_question_id": metacognition_guards["expected_question_id"],
            "expected_context_version": metacognition_guards["context_version"],
            "profile_revision": metacognition_guards["profile_summary"][
                "profile_revision"
            ],
            "learner_jol_percent": 80,
            "strategy_codes": ["self_explanation", "checking"],
        }
    )
    assert prediction["applied"] is True

    revoke_body = {
        "syllabus_id": syllabus["syllabus_id"],
        "curriculum_id": sealed["curriculum_blueprint"]["curriculum_id"],
        "reason_code": "teacher_revoked",
        "expected_syllabus_version": 6,
        "expected_authority_version": 2,
        "curriculum_authority_idempotency_key": "dashboard-revoke-0001",
    }
    snapshot.revoke_curriculum(
        _gateway_request(
            verifier,
            path="api/curriculum/revoke",
            body=revoke_body,
            nonce="1",
        )
    )
    with pytest.raises(
        TeacherAgentDashboardError, match="curriculum authority is no longer active"
    ):
        snapshot.step(
            {
                "session_id": started["session_id"],
                "expected_round": started["rounds_completed"],
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": started["context_version"],
                "profile_revision": started["profile_summary"]["profile_revision"],
                "idempotency_key": "revoked-active-step-0001",
                "learner_response": "植物需要光能、二氧化碳和水。",
                "signal": "correct",
                "signal_confidence": 0.9,
            }
        )
    assert snapshot.sessions[started["session_id"]].session["round"] == 0
    with pytest.raises(
        TeacherAgentDashboardError, match="curriculum authority is no longer active"
    ):
        snapshot.pair_metacognitive_outcome(
            {
                "session_id": started["session_id"],
                "expected_round": metacognition_guards["rounds_completed"],
                "expected_question_id": metacognition_guards[
                    "expected_question_id"
                ],
                "expected_context_version": metacognition_guards[
                    "context_version"
                ],
                "profile_revision": metacognition_guards["profile_summary"][
                    "profile_revision"
                ],
                "prediction_event_id": prediction["prediction_event_id"],
            }
        )
    with pytest.raises(TeacherAgentDashboardError, match="not actively sealed"):
        snapshot.syllabus_lesson_payload(syllabus["syllabus_id"], "lesson_01_01")

    trashed = snapshot.trash_project(project["project_id"], {})
    deletion = snapshot.purge_project(
        project["project_id"],
        {
            "recovery_token": trashed["recovery_token"],
            "confirmation": deletion_confirmation(project["project_id"]),
        },
    )["deletion_receipt"]
    assert deletion["deleted_counts"]["curriculum_authority_families"] == 1
    assert deletion["deleted_counts"]["curriculum_authority_events"] == 3
    assert family["family_id"] not in authority_store.path.read_text()
