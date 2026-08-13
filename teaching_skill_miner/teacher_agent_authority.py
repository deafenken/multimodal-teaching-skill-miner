"""Scope-bound deployment authorization for authenticated teacher mutations.

The gateway authenticates a principal and checks an exact role allowlist.  It
then signs a short-lived request envelope with a key derived from the private
tenant/owner worker scope.  This module verifies that service assertion and
records a hash-only replay fence before adjudication or resource-review mutation.

The receipts prove that this deployment authorized an already-authenticated
role.  They are deliberately not personal/non-repudiable teacher signatures.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Callable, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - production gateway is POSIX-only.
    fcntl = None  # type: ignore[assignment]


TEACHER_AUTHORITY_SCHEMA = "teaching_skill_miner.teacher_authority_envelope.v1"
TEACHER_AUTHORITY_RECEIPT_SCHEMA = (
    "teaching_skill_miner.teacher_authority_verification_receipt.v1"
)
TEACHER_AUTHORITY_REVALIDATION_SCHEMA = (
    "teaching_skill_miner.teacher_authority_revalidation_receipt.v1"
)
TEACHER_AUTHORITY_REPLAY_SCHEMA = (
    "teaching_skill_miner.teacher_authority_replay_event.v1"
)
AUTHENTICATED_TEACHER_ACTOR = "authenticated_teacher_server_authorized"
AUTHORITY_ASSURANCE = "deployment_service_role_authorization_not_personal_signature"
TEACHER_AUTHORITY_REPLAY_MAX_BYTES = 16 * 1024 * 1024

_DIGEST = re.compile(r"[0-9a-f]{64}")
_SCOPE = re.compile(r"scope_[0-9a-f]{48}")
_KEY_VERSION = re.compile(r"k[1-9][0-9]{0,8}")
_AUTHORITY_ID = re.compile(r"tauth_[0-9a-f]{24}")
_NONCE = re.compile(r"tan_[0-9a-f]{48}")
_SIGNATURE = re.compile(r"[A-Za-z0-9_-]{43}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}")
_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,159}")
_AUTHORITY_IDEMPOTENCY_FIELDS = {
    "api/adjudication/claim": "adjudication_idempotency_key",
    "api/adjudication/decide": "adjudication_idempotency_key",
    "api/resource/review": "resource_review_idempotency_key",
    "api/curriculum/review": "curriculum_authority_idempotency_key",
    "api/curriculum/seal": "curriculum_authority_idempotency_key",
    "api/curriculum/revoke": "curriculum_authority_idempotency_key",
    "api/safeguarding/list": "safeguarding_idempotency_key",
    "api/safeguarding/dispatch": "safeguarding_idempotency_key",
    "api/safeguarding/case/acknowledge": "safeguarding_idempotency_key",
    "api/safeguarding/case/close": "safeguarding_idempotency_key",
    "api/safeguarding/escalation/overdue": "safeguarding_idempotency_key",
    "api/safeguarding/escalation/acknowledge": "safeguarding_idempotency_key",
}
_ENVELOPE_KEYS = frozenset(
    {
        "schema",
        "authority_kind",
        "assurance",
        "scope_id",
        "scope_key_version",
        "actor_principal_sha256",
        "roles_sha256",
        "role_policy_sha256",
        "method",
        "path",
        "body_sha256",
        "idempotency_key_sha256",
        "issued_at",
        "expires_at",
        "nonce",
        "authority_id",
        "signature",
    }
)
_REPLAY_KEYS = frozenset(
    {
        "schema",
        "seq",
        "authority_id",
        "nonce_sha256",
        "consumed_at",
        "expires_at",
        "request_sha256",
        "previous_hash",
        "hash",
    }
)
_VERIFICATION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "authority_id",
        "authority_kind",
        "assurance",
        "scope_id",
        "scope_key_version",
        "actor_principal_sha256",
        "roles_sha256",
        "role_policy_sha256",
        "method",
        "path",
        "body_sha256",
        "idempotency_key_sha256",
        "issued_at",
        "expires_at",
        "nonce_sha256",
        "gateway_envelope_sha256",
        "service_authorization_signature_verified",
        "personal_non_repudiation",
        "signature",
        "receipt_sha256",
    }
)
_REVALIDATION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "authority_kind",
        "assurance",
        "authority_id",
        "gateway_verification_receipt_sha256",
        "scope_id",
        "scope_key_version",
        "actor_principal_sha256",
        "roles_sha256",
        "method",
        "path",
        "request_body_sha256",
        "idempotency_key_sha256",
        "review_item_id",
        "review_item_version_sha256",
        "evidence_id",
        "evidence_sha256",
        "model_evidence_sha256",
        "instruction_authority_basis_sha256",
        "correction_sha256",
        "target_kc_ids",
        "rubric_id",
        "rubric_authority_sha256",
        "revalidated_at",
        "personal_non_repudiation",
        "signature",
        "receipt_sha256",
    }
)


class TeacherAuthorityError(ValueError):
    """Raised when an authorization assertion is absent or untrusted."""


class TeacherAuthorityReplayError(TeacherAuthorityError):
    """Raised when an already consumed service authorization is replayed."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherAuthorityError(
            "teacher authority values must contain canonical JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise TeacherAuthorityError(f"teacher authority {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TeacherAuthorityError(f"teacher authority {field} is invalid") from exc
    if parsed.tzinfo is None:
        raise TeacherAuthorityError(f"teacher authority {field} is invalid")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise TeacherAuthorityError(f"teacher authority {field} is invalid")
    return value


def _b64url_signature(key: bytes, value: Mapping[str, Any]) -> str:
    return (
        base64.urlsafe_b64encode(hmac.new(key, canonical_bytes(value), sha256).digest())
        .decode("ascii")
        .rstrip("=")
    )


def validate_teacher_authority_verification_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(receipt, Mapping)
        or frozenset(receipt) != _VERIFICATION_RECEIPT_KEYS
        or receipt.get("schema") != TEACHER_AUTHORITY_RECEIPT_SCHEMA
        or receipt.get("authority_kind") != AUTHENTICATED_TEACHER_ACTOR
        or receipt.get("assurance") != AUTHORITY_ASSURANCE
        or receipt.get("service_authorization_signature_verified") is not True
        or receipt.get("personal_non_repudiation") is not False
    ):
        raise TeacherAuthorityError("teacher authority verification receipt is invalid")
    declared = _digest(receipt.get("receipt_sha256"), field="receipt_sha256")
    material = deepcopy(dict(receipt))
    material.pop("receipt_sha256")
    if not hmac.compare_digest(declared, canonical_sha256(material)):
        raise TeacherAuthorityError("teacher authority verification receipt is invalid")
    _authority_id(receipt.get("authority_id"))
    for field in (
        "actor_principal_sha256",
        "roles_sha256",
        "role_policy_sha256",
        "body_sha256",
        "idempotency_key_sha256",
        "nonce_sha256",
        "gateway_envelope_sha256",
    ):
        _digest(receipt.get(field), field=field)
    signature = receipt.get("signature")
    if not isinstance(signature, str) or _SIGNATURE.fullmatch(signature) is None:
        raise TeacherAuthorityError("teacher authority verification receipt is invalid")
    return deepcopy(dict(receipt))


def validate_teacher_authority_revalidation_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(receipt, Mapping)
        or frozenset(receipt) != _REVALIDATION_RECEIPT_KEYS
    ):
        raise TeacherAuthorityError("teacher authority revalidation is missing")
    original = deepcopy(dict(receipt))
    value = deepcopy(original)
    declared = value.pop("receipt_sha256", None)
    if not isinstance(declared, str) or not hmac.compare_digest(
        declared, canonical_sha256(value)
    ):
        raise TeacherAuthorityError("teacher authority receipt integrity failed")
    signature = value.get("signature")
    if not isinstance(signature, str) or _SIGNATURE.fullmatch(signature) is None:
        raise TeacherAuthorityError("teacher authority receipt signature is invalid")
    if (
        value.get("schema") != TEACHER_AUTHORITY_REVALIDATION_SCHEMA
        or value.get("authority_kind") != AUTHENTICATED_TEACHER_ACTOR
        or value.get("assurance") != AUTHORITY_ASSURANCE
        or value.get("personal_non_repudiation") is not False
    ):
        raise TeacherAuthorityError("teacher authority receipt binding failed")
    for field in (
        "gateway_verification_receipt_sha256",
        "actor_principal_sha256",
        "roles_sha256",
        "request_body_sha256",
        "idempotency_key_sha256",
        "review_item_version_sha256",
        "evidence_sha256",
        "model_evidence_sha256",
        "instruction_authority_basis_sha256",
        "correction_sha256",
        "rubric_authority_sha256",
    ):
        _digest(value.get(field), field=field)
    _authority_id(value.get("authority_id"))
    _timestamp(value.get("revalidated_at"), field="revalidated_at")
    if (
        value.get("method") != "POST"
        or value.get("path") != "api/adjudication/decide"
        or not isinstance(value.get("target_kc_ids"), list)
        or not value["target_kc_ids"]
        or any(
            not isinstance(kc_id, str) or _SAFE_ID.fullmatch(kc_id) is None
            for kc_id in value["target_kc_ids"]
        )
    ):
        raise TeacherAuthorityError("teacher authority receipt binding failed")
    return original


def authenticated_teacher_actor(receipt: Mapping[str, Any]) -> dict[str, Any]:
    receipt = validate_teacher_authority_verification_receipt(receipt)
    return {
        "identity": AUTHENTICATED_TEACHER_ACTOR,
        "authenticated": True,
        "teacher_identity_claimed": True,
        "principal_sha256": _digest(
            receipt.get("actor_principal_sha256"), field="actor_principal_sha256"
        ),
        "roles_sha256": _digest(receipt.get("roles_sha256"), field="roles_sha256"),
        "authority_id": _authority_id(receipt.get("authority_id")),
        "assurance": AUTHORITY_ASSURANCE,
    }


def _authority_id(value: Any) -> str:
    if not isinstance(value, str) or _AUTHORITY_ID.fullmatch(value) is None:
        raise TeacherAuthorityError("teacher authority ID is invalid")
    return value


class TeacherAuthorityVerifier:
    """Verify scope assertions and persist one-time nonce consumption."""

    def __init__(
        self,
        *,
        key: bytes,
        scope_id: str,
        scope_key_version: str,
        replay_store_path: str | Path,
        legacy_scope_bindings: Sequence[tuple[str, str]] = (),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise TeacherAuthorityError("teacher authority key is invalid")
        if _SCOPE.fullmatch(scope_id) is None:
            raise TeacherAuthorityError("teacher authority scope is invalid")
        if _KEY_VERSION.fullmatch(scope_key_version) is None:
            raise TeacherAuthorityError("teacher authority key version is invalid")
        self._key = key
        self.scope_id = scope_id
        self.scope_key_version = scope_key_version
        normalized_bindings = {(scope_id, scope_key_version)}
        if (
            not isinstance(legacy_scope_bindings, Sequence)
            or isinstance(legacy_scope_bindings, (str, bytes))
            or len(legacy_scope_bindings) > 16
        ):
            raise TeacherAuthorityError(
                "teacher authority legacy scope bindings are invalid"
            )
        for binding in legacy_scope_bindings:
            if (
                not isinstance(binding, tuple)
                or len(binding) != 2
                or _SCOPE.fullmatch(binding[0]) is None
                or _KEY_VERSION.fullmatch(binding[1]) is None
            ):
                raise TeacherAuthorityError(
                    "teacher authority legacy scope binding is invalid"
                )
            normalized_bindings.add(binding)
        self._receipt_scope_bindings = frozenset(normalized_bindings)
        self.path = Path(replay_store_path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()

    def verify(
        self,
        body: Mapping[str, Any],
        *,
        method: str,
        path: str,
        consume: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(body, Mapping):
            raise TeacherAuthorityError("teacher authority request must be an object")
        envelope_raw = body.get("_teacher_authority")
        if (
            not isinstance(envelope_raw, Mapping)
            or frozenset(envelope_raw) != _ENVELOPE_KEYS
        ):
            raise TeacherAuthorityError(
                "teacher authority envelope is missing or invalid"
            )
        envelope = deepcopy(dict(envelope_raw))
        if (
            envelope.get("schema") != TEACHER_AUTHORITY_SCHEMA
            or envelope.get("authority_kind") != AUTHENTICATED_TEACHER_ACTOR
            or envelope.get("assurance") != AUTHORITY_ASSURANCE
            or envelope.get("scope_id") != self.scope_id
            or envelope.get("scope_key_version") != self.scope_key_version
            or envelope.get("method") != method
            or envelope.get("path") != path
        ):
            raise TeacherAuthorityError("teacher authority envelope binding is invalid")
        for field in (
            "actor_principal_sha256",
            "roles_sha256",
            "role_policy_sha256",
            "body_sha256",
            "idempotency_key_sha256",
        ):
            _digest(envelope.get(field), field=field)
        authority_id = envelope.get("authority_id")
        nonce = envelope.get("nonce")
        signature = envelope.pop("signature", None)
        if (
            not isinstance(authority_id, str)
            or _AUTHORITY_ID.fullmatch(authority_id) is None
            or not isinstance(nonce, str)
            or _NONCE.fullmatch(nonce) is None
            or not isinstance(signature, str)
            or _SIGNATURE.fullmatch(signature) is None
        ):
            raise TeacherAuthorityError(
                "teacher authority envelope identity is invalid"
            )
        expected_authority_id = f"tauth_{canonical_sha256({key: value for key, value in envelope.items() if key != 'authority_id'})[:24]}"
        if not hmac.compare_digest(authority_id, expected_authority_id):
            raise TeacherAuthorityError("teacher authority ID binding is invalid")
        if not hmac.compare_digest(signature, _b64url_signature(self._key, envelope)):
            raise TeacherAuthorityError("teacher authority signature is invalid")

        request_body = deepcopy(dict(body))
        request_body.pop("_teacher_authority", None)
        request_hash = canonical_sha256(request_body)
        if not hmac.compare_digest(str(envelope["body_sha256"]), request_hash):
            raise TeacherAuthorityError("teacher authority request body changed")
        idempotency_field = _AUTHORITY_IDEMPOTENCY_FIELDS.get(path)
        if idempotency_field is None:
            raise TeacherAuthorityError("teacher authority operation is unsupported")
        idempotency_key = request_body.get(idempotency_field)
        if (
            not isinstance(idempotency_key, str)
            or _IDEMPOTENCY_KEY.fullmatch(idempotency_key) is None
            or not hmac.compare_digest(
                str(envelope["idempotency_key_sha256"]),
                sha256(idempotency_key.encode("utf-8")).hexdigest(),
            )
        ):
            raise TeacherAuthorityError(
                "teacher authority idempotency binding is invalid"
            )

        now = self._clock().astimezone(timezone.utc)
        issued_at = _timestamp(envelope.get("issued_at"), field="issued_at")
        expires_at = _timestamp(envelope.get("expires_at"), field="expires_at")
        if issued_at > now + timedelta(seconds=30):
            raise TeacherAuthorityError("teacher authority was issued in the future")
        if expires_at <= now:
            raise TeacherAuthorityError("teacher authority has expired")
        if not timedelta(seconds=30) <= expires_at - issued_at <= timedelta(minutes=10):
            raise TeacherAuthorityError("teacher authority lifetime is invalid")

        envelope_with_signature = {**envelope, "signature": signature}
        verification: dict[str, Any] = {
            "schema": TEACHER_AUTHORITY_RECEIPT_SCHEMA,
            "authority_id": authority_id,
            "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
            "assurance": AUTHORITY_ASSURANCE,
            "scope_id": self.scope_id,
            "scope_key_version": self.scope_key_version,
            "actor_principal_sha256": envelope["actor_principal_sha256"],
            "roles_sha256": envelope["roles_sha256"],
            "role_policy_sha256": envelope["role_policy_sha256"],
            "method": method,
            "path": path,
            "body_sha256": request_hash,
            "idempotency_key_sha256": envelope["idempotency_key_sha256"],
            "issued_at": envelope["issued_at"],
            "expires_at": envelope["expires_at"],
            "nonce_sha256": sha256(nonce.encode("ascii")).hexdigest(),
            "gateway_envelope_sha256": canonical_sha256(envelope_with_signature),
            "service_authorization_signature_verified": True,
            "personal_non_repudiation": False,
        }
        verification["signature"] = _b64url_signature(self._key, verification)
        verification["receipt_sha256"] = canonical_sha256(verification)
        if consume:
            self._consume(verification, now=now)
        return verification

    def verify_verification_receipt(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        value = validate_teacher_authority_verification_receipt(receipt)
        value.pop("receipt_sha256")
        signature = value.pop("signature")
        if not hmac.compare_digest(str(signature), _b64url_signature(self._key, value)):
            raise TeacherAuthorityError(
                "teacher authority verification receipt signature failed"
            )
        if (
            value.get("scope_id"),
            value.get("scope_key_version"),
        ) not in self._receipt_scope_bindings:
            raise TeacherAuthorityError(
                "teacher authority verification receipt scope failed"
            )
        return deepcopy(dict(receipt))

    def revalidate(
        self,
        verification: Mapping[str, Any],
        *,
        review_item_id: str,
        review_item_version_sha256: str,
        evidence_id: str,
        evidence_sha256: str,
        model_evidence_sha256: str,
        instruction_authority_basis_sha256: str,
        correction: Mapping[str, Any],
        target_kc_ids: Sequence[str],
        rubric_id: str,
        rubric_authority_sha256: str,
    ) -> dict[str, Any]:
        for value, pattern, field in (
            (review_item_id, re.compile(r"adj_[0-9a-f]{24}"), "review_item_id"),
            (evidence_id, _SAFE_ID, "evidence_id"),
            (rubric_id, _SAFE_ID, "rubric_id"),
        ):
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise TeacherAuthorityError(f"teacher authority {field} is invalid")
        for value, field in (
            (review_item_version_sha256, "review_item_version_sha256"),
            (evidence_sha256, "evidence_sha256"),
            (model_evidence_sha256, "model_evidence_sha256"),
            (
                instruction_authority_basis_sha256,
                "instruction_authority_basis_sha256",
            ),
            (rubric_authority_sha256, "rubric_authority_sha256"),
        ):
            _digest(value, field=field)
        if (
            not isinstance(correction, Mapping)
            or not target_kc_ids
            or any(not isinstance(kc_id, str) for kc_id in target_kc_ids)
        ):
            raise TeacherAuthorityError(
                "teacher authority correction binding is invalid"
            )
        receipt: dict[str, Any] = {
            "schema": TEACHER_AUTHORITY_REVALIDATION_SCHEMA,
            "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
            "assurance": AUTHORITY_ASSURANCE,
            "authority_id": verification["authority_id"],
            "gateway_verification_receipt_sha256": verification["receipt_sha256"],
            "scope_id": self.scope_id,
            "scope_key_version": self.scope_key_version,
            "actor_principal_sha256": verification["actor_principal_sha256"],
            "roles_sha256": verification["roles_sha256"],
            "method": verification["method"],
            "path": verification["path"],
            "request_body_sha256": verification["body_sha256"],
            "idempotency_key_sha256": verification["idempotency_key_sha256"],
            "review_item_id": review_item_id,
            "review_item_version_sha256": review_item_version_sha256,
            "evidence_id": evidence_id,
            "evidence_sha256": evidence_sha256,
            "model_evidence_sha256": model_evidence_sha256,
            "instruction_authority_basis_sha256": (instruction_authority_basis_sha256),
            "correction_sha256": canonical_sha256(dict(correction)),
            "target_kc_ids": list(target_kc_ids),
            "rubric_id": rubric_id,
            "rubric_authority_sha256": rubric_authority_sha256,
            "revalidated_at": _format_timestamp(self._clock()),
            "personal_non_repudiation": False,
        }
        receipt["signature"] = _b64url_signature(self._key, receipt)
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return receipt

    def verify_revalidation(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        original = validate_teacher_authority_revalidation_receipt(receipt)
        value = deepcopy(original)
        value.pop("receipt_sha256")
        signature = value.pop("signature", None)
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, _b64url_signature(self._key, value)
        ):
            raise TeacherAuthorityError("teacher authority receipt signature failed")
        if (
            value.get("scope_id"),
            value.get("scope_key_version"),
        ) not in self._receipt_scope_bindings:
            raise TeacherAuthorityError("teacher authority receipt binding failed")
        return original

    def _consume(self, verification: Mapping[str, Any], *, now: datetime) -> None:
        if fcntl is None:
            raise TeacherAuthorityError("teacher authority replay store requires POSIX")
        with self._lock:
            if not hasattr(os, "O_NOFOLLOW"):
                raise TeacherAuthorityError(
                    "teacher authority replay store requires no-follow opens"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            parent = self.path.parent.lstat()
            if (
                stat.S_ISLNK(parent.st_mode)
                or not stat.S_ISDIR(parent.st_mode)
                or parent.st_mode & 0o077
                or (hasattr(os, "getuid") and parent.st_uid != os.getuid())
            ):
                raise TeacherAuthorityError(
                    "teacher authority replay store directory is unsafe"
                )
            descriptor = os.open(
                self.path,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
                ):
                    raise TeacherAuthorityError(
                        "teacher authority replay store is unsafe"
                    )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                os.lseek(descriptor, 0, os.SEEK_SET)
                raw = b""
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    raw += chunk
                    if len(raw) > TEACHER_AUTHORITY_REPLAY_MAX_BYTES:
                        raise TeacherAuthorityError(
                            "teacher authority replay store is too large"
                        )
                events: list[dict[str, Any]] = []
                previous_hash: str | None = None
                seen: set[str] = set()
                if raw and not raw.endswith(b"\n"):
                    raise TeacherAuthorityError(
                        "teacher authority replay store is torn"
                    )
                for index, line in enumerate(raw.splitlines(), start=1):
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise TeacherAuthorityError(
                            "teacher authority replay store is invalid"
                        ) from exc
                    if not isinstance(event, dict) or frozenset(event) != _REPLAY_KEYS:
                        raise TeacherAuthorityError(
                            "teacher authority replay event is invalid"
                        )
                    declared = event.get("hash")
                    material = deepcopy(event)
                    material.pop("hash", None)
                    if (
                        event.get("schema") != TEACHER_AUTHORITY_REPLAY_SCHEMA
                        or event.get("seq") != index
                        or event.get("previous_hash") != previous_hash
                        or not isinstance(declared, str)
                        or not hmac.compare_digest(declared, canonical_sha256(material))
                    ):
                        raise TeacherAuthorityError(
                            "teacher authority replay chain is invalid"
                        )
                    nonce_hash = _digest(
                        event.get("nonce_sha256"), field="nonce_sha256"
                    )
                    if nonce_hash in seen:
                        raise TeacherAuthorityError(
                            "teacher authority replay chain duplicates a nonce"
                        )
                    seen.add(nonce_hash)
                    events.append(event)
                    previous_hash = declared
                nonce_hash = str(verification["nonce_sha256"])
                if nonce_hash in seen:
                    raise TeacherAuthorityReplayError(
                        "teacher authority envelope was replayed"
                    )
                event: dict[str, Any] = {
                    "schema": TEACHER_AUTHORITY_REPLAY_SCHEMA,
                    "seq": len(events) + 1,
                    "authority_id": verification["authority_id"],
                    "nonce_sha256": nonce_hash,
                    "consumed_at": _format_timestamp(now),
                    "expires_at": verification["expires_at"],
                    "request_sha256": verification["body_sha256"],
                    "previous_hash": previous_hash,
                }
                event["hash"] = canonical_sha256(event)
                os.lseek(descriptor, 0, os.SEEK_END)
                payload = canonical_bytes(event) + b"\n"
                if len(raw) + len(payload) > TEACHER_AUTHORITY_REPLAY_MAX_BYTES:
                    raise TeacherAuthorityError(
                        "teacher authority replay store capacity is exhausted"
                    )
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise TeacherAuthorityError(
                            "teacher authority replay store write failed"
                        )
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)


__all__ = [
    "AUTHENTICATED_TEACHER_ACTOR",
    "AUTHORITY_ASSURANCE",
    "TEACHER_AUTHORITY_RECEIPT_SCHEMA",
    "TEACHER_AUTHORITY_REVALIDATION_SCHEMA",
    "TEACHER_AUTHORITY_REPLAY_MAX_BYTES",
    "TEACHER_AUTHORITY_SCHEMA",
    "TeacherAuthorityError",
    "TeacherAuthorityReplayError",
    "TeacherAuthorityVerifier",
    "authenticated_teacher_actor",
    "canonical_sha256",
    "validate_teacher_authority_verification_receipt",
    "validate_teacher_authority_revalidation_receipt",
]
