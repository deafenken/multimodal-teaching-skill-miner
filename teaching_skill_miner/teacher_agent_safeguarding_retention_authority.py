"""Restart-safe server authority for safeguarding retention compaction.

This authority is intentionally independent from both the safety classifier
and authenticated safeguarding staff.  It can authorize one exact, hash-only
retention plan and no case mutation.  A keyed proof makes verification durable
across process restarts without retaining a registry of issued receipts.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
from typing import Any, Callable

from .teacher_agent_safeguarding import (
    SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA,
    SAFEGUARDING_RETENTION_ROLE_SHA256,
    canonical_sha256,
    retention_authorization_receipt_sha256,
)


class SafeguardingRetentionAuthorityError(RuntimeError):
    """Raised when the dedicated retention authority cannot prove a receipt."""


_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "actor_kind",
        "server_authenticated",
        "principal_sha256",
        "role_sha256",
        "authorization_context_sha256",
        "operation",
        "body_sha256",
        "issued_at_utc",
        "expires_at_utc",
        "authority_proof_sha256",
        "receipt_sha256",
    }
)
_DIGEST_ALPHABET = frozenset("0123456789abcdef")
_OPERATION = "case.retention_compacted"


def _digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_ALPHABET for character in value)
    ):
        raise SafeguardingRetentionAuthorityError(f"{field} is invalid")
    return value


def _utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SafeguardingRetentionAuthorityError(f"{field} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SafeguardingRetentionAuthorityError(f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        raise SafeguardingRetentionAuthorityError(f"{field} is invalid")
    return parsed.astimezone(timezone.utc)


class InternalSafeguardingRetentionAuthority:
    """Issue HMAC-bound receipts for one exact retention operation body.

    ``key`` is a deployment-only secret containing at least 32 random bytes.
    Reconstructing the authority with the same key and deployment-context hash
    verifies unexpired receipts after restart.  Neither value is written to the
    safeguarding store.
    """

    def __init__(
        self,
        *,
        key: bytes,
        deployment_context_sha256: str,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not isinstance(key, bytes) or len(key) < 32:
            raise SafeguardingRetentionAuthorityError(
                "retention authority key must contain at least 32 bytes"
            )
        _digest(
            deployment_context_sha256,
            field="retention authority deployment context",
        )
        if not callable(clock):
            raise SafeguardingRetentionAuthorityError(
                "retention authority clock is invalid"
            )
        self._key = key
        self._deployment_context_sha256 = deployment_context_sha256
        self._clock = clock

    def _proof(self, value: Mapping[str, Any]) -> str:
        material = deepcopy(dict(value))
        material.pop("authority_proof_sha256", None)
        material.pop("receipt_sha256", None)
        material_digest = bytes.fromhex(canonical_sha256(material))
        return hmac.new(
            self._key,
            b"safeguarding-retention-authority-receipt-v1\x00" + material_digest,
            sha256,
        ).hexdigest()

    def issue(
        self,
        *,
        operation: str,
        body_sha256: str,
        lifetime_seconds: int = 300,
    ) -> dict[str, Any]:
        if operation != _OPERATION:
            raise SafeguardingRetentionAuthorityError(
                "retention authority operation is invalid"
            )
        _digest(body_sha256, field="retention authority body_sha256")
        if (
            isinstance(lifetime_seconds, bool)
            or not isinstance(lifetime_seconds, int)
            or not 1 <= lifetime_seconds <= 15 * 60
        ):
            raise SafeguardingRetentionAuthorityError(
                "retention authority receipt lifetime is invalid"
            )
        now_value = self._clock()
        if not isinstance(now_value, datetime) or now_value.tzinfo is None:
            raise SafeguardingRetentionAuthorityError(
                "retention authority clock is invalid"
            )
        now = now_value.astimezone(timezone.utc).replace(microsecond=0)
        expires = now + timedelta(seconds=lifetime_seconds)
        receipt = {
            "schema": SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA,
            "version": 1,
            "actor_kind": "server_safeguarding_retention",
            "server_authenticated": True,
            "principal_sha256": hmac.new(
                self._key, b"safeguarding-retention-principal-v1", sha256
            ).hexdigest(),
            "role_sha256": SAFEGUARDING_RETENTION_ROLE_SHA256,
            "authorization_context_sha256": hmac.new(
                self._key,
                (
                    b"safeguarding-retention-deployment-context-v1\x00"
                    + self._deployment_context_sha256.encode("ascii")
                ),
                sha256,
            ).hexdigest(),
            "operation": operation,
            "body_sha256": body_sha256,
            "issued_at_utc": now.isoformat().replace("+00:00", "Z"),
            "expires_at_utc": expires.isoformat().replace("+00:00", "Z"),
        }
        receipt["authority_proof_sha256"] = self._proof(receipt)
        receipt["receipt_sha256"] = retention_authorization_receipt_sha256(receipt)
        return deepcopy(receipt)

    def verify(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping) or frozenset(value) != _RECEIPT_KEYS:
            raise SafeguardingRetentionAuthorityError(
                "retention authority receipt shape is invalid"
            )
        receipt = deepcopy(dict(value))
        if (
            receipt.get("schema") != SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA
            or type(receipt.get("version")) is not int
            or receipt.get("version") != 1
            or receipt.get("actor_kind") != "server_safeguarding_retention"
            or receipt.get("server_authenticated") is not True
            or receipt.get("role_sha256") != SAFEGUARDING_RETENTION_ROLE_SHA256
            or receipt.get("operation") != _OPERATION
        ):
            raise SafeguardingRetentionAuthorityError(
                "retention authority receipt policy is invalid"
            )
        for field in (
            "principal_sha256",
            "role_sha256",
            "authorization_context_sha256",
            "body_sha256",
            "authority_proof_sha256",
            "receipt_sha256",
        ):
            _digest(receipt.get(field), field=f"retention receipt {field}")
        issued = _utc(receipt.get("issued_at_utc"), field="retention issued_at_utc")
        expires = _utc(receipt.get("expires_at_utc"), field="retention expires_at_utc")
        now_value = self._clock()
        if not isinstance(now_value, datetime) or now_value.tzinfo is None:
            raise SafeguardingRetentionAuthorityError(
                "retention authority clock is invalid"
            )
        now = now_value.astimezone(timezone.utc)
        expected_principal = hmac.new(
            self._key, b"safeguarding-retention-principal-v1", sha256
        ).hexdigest()
        expected_context = hmac.new(
            self._key,
            (
                b"safeguarding-retention-deployment-context-v1\x00"
                + self._deployment_context_sha256.encode("ascii")
            ),
            sha256,
        ).hexdigest()
        if (
            expires <= now
            or expires <= issued
            or (expires - issued).total_seconds() > 15 * 60
            or issued > now + timedelta(seconds=60)
            or not hmac.compare_digest(
                str(receipt["principal_sha256"]), expected_principal
            )
            or not hmac.compare_digest(
                str(receipt["authorization_context_sha256"]), expected_context
            )
            or not hmac.compare_digest(
                str(receipt["authority_proof_sha256"]), self._proof(receipt)
            )
            or not hmac.compare_digest(
                str(receipt["receipt_sha256"]),
                retention_authorization_receipt_sha256(receipt),
            )
        ):
            raise SafeguardingRetentionAuthorityError(
                "retention authority receipt verification failed"
            )
        return receipt


__all__ = [
    "InternalSafeguardingRetentionAuthority",
    "SafeguardingRetentionAuthorityError",
]
