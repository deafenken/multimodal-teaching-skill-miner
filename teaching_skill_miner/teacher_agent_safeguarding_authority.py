"""Private in-process authorities for safeguarding receipts.

The safety classifier and safeguarding staff deliberately use separate issuer
registries.  A classifier can only open a case.  A staff receipt can only be
minted after the scope worker has verified an Apps-gateway authority envelope
whose exact route policy is the single ``safeguarding`` role.  Neither receipt
is a browser-computable bearer assertion: the public hash is accepted only
while it is present in the corresponding bounded private mint registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import threading
from typing import Any, Callable

from .teacher_agent_safeguarding import (
    SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA,
    SAFEGUARDING_ROLE_SHA256,
    SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256,
    authorization_receipt_sha256,
)
from .teacher_agent_authority import (
    TEACHER_AUTHORITY_RECEIPT_SCHEMA,
    canonical_sha256 as teacher_authority_sha256,
)


class SafeguardingSystemAuthorityError(RuntimeError):
    """Raised when a receipt was not minted by this private worker."""


class SafeguardingStaffAuthorityError(RuntimeError):
    """Raised when fresh gateway safeguarding authority cannot be proven."""


_STAFF_OPERATION_PATHS = {
    "case.acknowledged": "api/safeguarding/case/acknowledge",
    "case.closed": "api/safeguarding/case/close",
    "escalation.overdue": "api/safeguarding/escalation/overdue",
    "escalation.acknowledged": "api/safeguarding/escalation/acknowledge",
}
SAFEGUARDING_GATEWAY_ROLE_POLICY_SHA256 = teacher_authority_sha256(
    ["safeguarding"]
)


class InternalSafeguardingSystemAuthority:
    """Mint and verify short-lived, hash-only receipts inside one scope worker.

    The public receipt intentionally has no bearer signature field accepted by
    the durable store.  Authenticity is instead proven by membership in this
    private issuer's bounded in-memory mint registry, so a browser cannot forge
    a classifier identity by recomputing public JSON hashes.
    """

    def __init__(
        self,
        *,
        key: bytes,
        scope_sha256: str,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not isinstance(key, bytes) or len(key) < 32:
            raise SafeguardingSystemAuthorityError("authority key is invalid")
        if (
            not isinstance(scope_sha256, str)
            or len(scope_sha256) != 64
            or any(character not in "0123456789abcdef" for character in scope_sha256)
        ):
            raise SafeguardingSystemAuthorityError("authority scope is invalid")
        if not callable(clock):
            raise SafeguardingSystemAuthorityError("authority clock is invalid")
        self._key = key
        self._scope_sha256 = scope_sha256
        self._clock = clock
        self._minted: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def issue(self, *, operation: str, body_sha256: str) -> dict[str, Any]:
        if operation != "case.opened" or not isinstance(body_sha256, str):
            raise SafeguardingSystemAuthorityError("authority request is invalid")
        now = self._clock().astimezone(timezone.utc).replace(microsecond=0)
        expires = now + timedelta(minutes=5)
        issued_at = now.isoformat().replace("+00:00", "Z")
        expires_at = expires.isoformat().replace("+00:00", "Z")
        receipt = {
            "schema": SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA,
            "actor_kind": "system_safety_classifier",
            "server_authenticated": True,
            "principal_sha256": hmac.new(
                self._key, b"system-safety-classifier", sha256
            ).hexdigest(),
            "role_sha256": SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256,
            "authorization_context_sha256": hmac.new(
                self._key,
                b"safeguarding-scope\x00" + self._scope_sha256.encode("ascii"),
                sha256,
            ).hexdigest(),
            "operation": operation,
            "body_sha256": body_sha256,
            "issued_at_utc": issued_at,
            "expires_at_utc": expires_at,
        }
        receipt["receipt_sha256"] = authorization_receipt_sha256(receipt)
        with self._lock:
            self._minted = {
                digest: expiry
                for digest, expiry in self._minted.items()
                if expiry > now
            }
            if len(self._minted) >= 4096:
                raise SafeguardingSystemAuthorityError(
                    "authority receipt capacity is exhausted"
                )
            self._minted[str(receipt["receipt_sha256"])] = expires
        return deepcopy(receipt)

    def verify(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SafeguardingSystemAuthorityError("authority receipt is invalid")
        receipt = deepcopy(dict(value))
        declared = receipt.get("receipt_sha256")
        if not isinstance(declared, str):
            raise SafeguardingSystemAuthorityError("authority receipt is invalid")
        computed = authorization_receipt_sha256(receipt)
        now = self._clock().astimezone(timezone.utc)
        with self._lock:
            expiry = self._minted.get(declared)
        if (
            expiry is None
            or expiry <= now
            or not hmac.compare_digest(declared, computed)
        ):
            raise SafeguardingSystemAuthorityError(
                "authority receipt was not minted by this worker"
            )
        return receipt


class InternalSafeguardingStaffAuthority:
    """Convert a verified exact-route gateway receipt into a core receipt.

    ``gateway_receipt_verifier`` must cryptographically re-verify the private
    scope worker receipt.  Cookie/session roles are never an input.  The role
    policy digest must equal the canonical singleton ``safeguarding`` role,
    which prevents a fresh but unrelated teacher entitlement from being
    upgraded inside the worker.
    """

    def __init__(
        self,
        *,
        key: bytes,
        scope_sha256: str,
        gateway_receipt_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not isinstance(key, bytes) or len(key) < 32:
            raise SafeguardingStaffAuthorityError("staff authority key is invalid")
        if (
            not isinstance(scope_sha256, str)
            or len(scope_sha256) != 64
            or any(character not in "0123456789abcdef" for character in scope_sha256)
        ):
            raise SafeguardingStaffAuthorityError("staff authority scope is invalid")
        if not callable(gateway_receipt_verifier) or not callable(clock):
            raise SafeguardingStaffAuthorityError(
                "staff authority verifier configuration is invalid"
            )
        self._key = key
        self._scope_sha256 = scope_sha256
        self._gateway_receipt_verifier = gateway_receipt_verifier
        self._clock = clock
        self._minted: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def issue(
        self,
        *,
        gateway_authorization_receipt: Mapping[str, Any],
        operation: str,
        body_sha256: str,
    ) -> dict[str, Any]:
        expected_path = _STAFF_OPERATION_PATHS.get(operation)
        if (
            expected_path is None
            or not isinstance(body_sha256, str)
            or len(body_sha256) != 64
            or any(character not in "0123456789abcdef" for character in body_sha256)
        ):
            raise SafeguardingStaffAuthorityError(
                "staff safeguarding authority request is invalid"
            )
        try:
            verified = self._gateway_receipt_verifier(
                deepcopy(dict(gateway_authorization_receipt))
            )
        except Exception as exc:
            raise SafeguardingStaffAuthorityError(
                "gateway safeguarding authority was rejected"
            ) from exc
        if (
            not isinstance(verified, Mapping)
            or verified.get("schema") != TEACHER_AUTHORITY_RECEIPT_SCHEMA
            or verified.get("method") != "POST"
            or verified.get("path") != expected_path
            or verified.get("role_policy_sha256")
            != SAFEGUARDING_GATEWAY_ROLE_POLICY_SHA256
            or verified.get("service_authorization_signature_verified") is not True
        ):
            raise SafeguardingStaffAuthorityError(
                "gateway authority is not the exact safeguarding operation"
            )
        principal_sha256 = verified.get("actor_principal_sha256")
        gateway_receipt_sha256 = verified.get("receipt_sha256")
        if (
            not isinstance(principal_sha256, str)
            or len(principal_sha256) != 64
            or not isinstance(gateway_receipt_sha256, str)
            or len(gateway_receipt_sha256) != 64
        ):
            raise SafeguardingStaffAuthorityError(
                "gateway safeguarding authority receipt is invalid"
            )
        now = self._clock().astimezone(timezone.utc).replace(microsecond=0)
        gateway_expiry_raw = verified.get("expires_at")
        try:
            gateway_expiry = datetime.fromisoformat(
                str(gateway_expiry_raw).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except (TypeError, ValueError) as exc:
            raise SafeguardingStaffAuthorityError(
                "gateway safeguarding authority expiry is invalid"
            ) from exc
        expires = min(now + timedelta(minutes=5), gateway_expiry)
        if expires <= now:
            raise SafeguardingStaffAuthorityError(
                "gateway safeguarding authority has expired"
            )
        issued_at = now.isoformat().replace("+00:00", "Z")
        expires_at = expires.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        receipt = {
            "schema": SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA,
            "actor_kind": "server_authenticated_safeguarding",
            "server_authenticated": True,
            "principal_sha256": principal_sha256,
            "role_sha256": SAFEGUARDING_ROLE_SHA256,
            "authorization_context_sha256": hmac.new(
                self._key,
                (
                    b"safeguarding-staff-context-v1\x00"
                    + self._scope_sha256.encode("ascii")
                    + b"\x00"
                    + gateway_receipt_sha256.encode("ascii")
                ),
                sha256,
            ).hexdigest(),
            "operation": operation,
            "body_sha256": body_sha256,
            "issued_at_utc": issued_at,
            "expires_at_utc": expires_at,
        }
        receipt["receipt_sha256"] = authorization_receipt_sha256(receipt)
        with self._lock:
            self._minted = {
                digest: expiry
                for digest, expiry in self._minted.items()
                if expiry > now
            }
            if len(self._minted) >= 4096:
                raise SafeguardingStaffAuthorityError(
                    "staff authority receipt capacity is exhausted"
                )
            self._minted[str(receipt["receipt_sha256"])] = expires
        return deepcopy(receipt)

    def verify(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SafeguardingStaffAuthorityError("staff authority receipt is invalid")
        receipt = deepcopy(dict(value))
        declared = receipt.get("receipt_sha256")
        if (
            receipt.get("actor_kind") != "server_authenticated_safeguarding"
            or receipt.get("role_sha256") != SAFEGUARDING_ROLE_SHA256
            or not isinstance(declared, str)
        ):
            raise SafeguardingStaffAuthorityError("staff authority receipt is invalid")
        computed = authorization_receipt_sha256(receipt)
        now = self._clock().astimezone(timezone.utc)
        with self._lock:
            expiry = self._minted.get(declared)
        if (
            expiry is None
            or expiry <= now
            or not hmac.compare_digest(declared, computed)
        ):
            raise SafeguardingStaffAuthorityError(
                "staff authority receipt was not minted by this worker"
            )
        return receipt


class CompositeSafeguardingAuthorizationVerifier:
    """Route receipts to the disjoint classifier and staff registries."""

    def __init__(
        self,
        *,
        system: InternalSafeguardingSystemAuthority,
        staff: InternalSafeguardingStaffAuthority,
    ) -> None:
        if not isinstance(system, InternalSafeguardingSystemAuthority) or not isinstance(
            staff, InternalSafeguardingStaffAuthority
        ):
            raise SafeguardingStaffAuthorityError(
                "composite safeguarding authority is invalid"
            )
        self._system = system
        self._staff = staff

    def verify(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SafeguardingStaffAuthorityError("safeguarding receipt is invalid")
        if value.get("actor_kind") == "system_safety_classifier":
            return self._system.verify(value)
        if value.get("actor_kind") == "server_authenticated_safeguarding":
            return self._staff.verify(value)
        raise SafeguardingStaffAuthorityError("safeguarding receipt actor is invalid")


__all__ = [
    "CompositeSafeguardingAuthorizationVerifier",
    "InternalSafeguardingStaffAuthority",
    "InternalSafeguardingSystemAuthority",
    "SAFEGUARDING_GATEWAY_ROLE_POLICY_SHA256",
    "SafeguardingStaffAuthorityError",
    "SafeguardingSystemAuthorityError",
]
