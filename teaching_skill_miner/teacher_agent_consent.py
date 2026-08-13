"""Durable, server-minted consent receipts for remote teaching effects.

Client booleans are requests, never proof of consent.  This store records the
specific purpose, provider, data classes, region, retention statement and
policy version approved by one opaque server-side subject.  Receipts contain no
learner text and can be revoked without deleting the audit trail.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

try:  # pragma: no cover - Windows uses the in-process lock only.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


CONSENT_STORE_SCHEMA = "teaching_skill_miner.remote_consent_store.v1"
CONSENT_RECEIPT_SCHEMA = "teaching_skill_miner.remote_consent_receipt.v1"
CONSENT_POLICY_VERSION = 1
MAX_CONSENT_EVENTS = 4096

_PURPOSES = frozenset(
    {
        "remote_chat",
        "remote_teaching",
        "remote_syllabus_generation",
        "public_web_search",
        "remote_visual_analysis",
    }
)
_DATA_CATEGORIES = frozenset(
    {
        "learner_message",
        "learner_profile_bounded",
        "teaching_resource_excerpt",
        "public_web_query",
        "learner_image",
    }
)
_GUARDIAN_POLICIES = frozenset(
    {"not_required", "verified_guardian", "verified_school_policy"}
)
_OPAQUE = re.compile(r"^[A-Za-z0-9_-]{16,160}$")
_PROVIDER = re.compile(r"^[A-Za-z0-9_.-]{2,80}$")
_REGION = re.compile(r"^[A-Za-z0-9_.-]{2,40}$")
_POLICY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,119}$")
_PROVIDER_POLICY_SOURCES = frozenset(
    {
        "deployment_operator_asserted_external_terms_not_repository_verified",
        "local_unverified_test_or_standalone",
    }
)
_PROVIDER_DELETION_STATUSES = frozenset(
    {
        "outside_service_control_subject_to_provider_policy",
        "provider_documents_zero_retention",
    }
)
_SUBJECT_POLICY_SOURCES = frozenset(
    {
        "organization_oidc_or_roster_policy",
        "local_unverified_self_declaration",
    }
)


class ConsentError(ValueError):
    """Raised when consent is absent, invalid, expired, or revoked."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ConsentError("consent clock must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ConsentError("consent timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ConsentError("consent timestamp is invalid") from exc
    return parsed.astimezone(timezone.utc)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConsentError("consent value is not canonical JSON") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_secret(secret: bytes) -> bytes:
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ConsentError("consent signing secret must contain at least 32 bytes")
    return secret


def _validated_https_policy_url(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise ConsentError("provider policy URL is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ConsentError("provider policy URL is invalid")
    return value


def _validated_provider_policy(
    raw: Mapping[str, Any] | None,
    *,
    provider_id: str,
    processing_region: str,
    provider_retention_days: int,
) -> dict[str, Any]:
    policy = (
        dict(raw)
        if isinstance(raw, Mapping)
        else {
            "policy_id": f"{provider_id}-local-unverified",
            "policy_version": "local-v1",
            "policy_source": "local_unverified_test_or_standalone",
            "processing_region": processing_region,
            "provider_retention_days": provider_retention_days,
            "deletion_status": "outside_service_control_subject_to_provider_policy",
            "documentation_url": None,
        }
    )
    if set(policy) != {
        "policy_id",
        "policy_version",
        "policy_source",
        "processing_region",
        "provider_retention_days",
        "deletion_status",
        "documentation_url",
    }:
        raise ConsentError("provider policy fields are invalid")
    if not _POLICY_COMPONENT.fullmatch(str(policy.get("policy_id", ""))):
        raise ConsentError("provider policy id is invalid")
    if not _POLICY_COMPONENT.fullmatch(str(policy.get("policy_version", ""))):
        raise ConsentError("provider policy version is invalid")
    if policy.get("policy_source") not in _PROVIDER_POLICY_SOURCES:
        raise ConsentError("provider policy source is invalid")
    if (
        policy.get("processing_region") != processing_region
        or policy.get("provider_retention_days") != provider_retention_days
    ):
        raise ConsentError("provider policy conflicts with consent processing terms")
    if policy.get("deletion_status") not in _PROVIDER_DELETION_STATUSES:
        raise ConsentError("provider deletion status is invalid")
    policy["documentation_url"] = _validated_https_policy_url(
        policy.get("documentation_url")
    )
    return policy


def _validated_subject_policy(
    raw: Mapping[str, Any] | None,
    *,
    likely_minor: bool,
    guardian_or_school_policy: str,
) -> dict[str, Any]:
    policy = (
        dict(raw)
        if isinstance(raw, Mapping)
        else {
            "policy_id": "local-self-declared-age-policy",
            "policy_version": "local-v1",
            "policy_source": "local_unverified_self_declaration",
            "likely_minor": likely_minor,
            "guardian_or_school_policy": guardian_or_school_policy,
            "remote_processing_eligible": True,
        }
    )
    if set(policy) != {
        "policy_id",
        "policy_version",
        "policy_source",
        "likely_minor",
        "guardian_or_school_policy",
        "remote_processing_eligible",
    }:
        raise ConsentError("subject policy fields are invalid")
    if not _POLICY_COMPONENT.fullmatch(str(policy.get("policy_id", ""))):
        raise ConsentError("subject policy id is invalid")
    if not _POLICY_COMPONENT.fullmatch(str(policy.get("policy_version", ""))):
        raise ConsentError("subject policy version is invalid")
    if policy.get("policy_source") not in _SUBJECT_POLICY_SOURCES:
        raise ConsentError("subject policy source is invalid")
    eligible = policy.get("remote_processing_eligible")
    if (
        policy.get("likely_minor") is not likely_minor
        or policy.get("guardian_or_school_policy") != guardian_or_school_policy
        or not isinstance(eligible, bool)
    ):
        raise ConsentError("subject policy does not authorize this consent")
    if likely_minor and eligible and guardian_or_school_policy == "not_required":
        raise ConsentError("minor remote processing lacks an authoritative policy")
    if not likely_minor and guardian_or_school_policy != "not_required":
        raise ConsentError("adult subject policy cannot claim guardian authorization")
    return policy


def validated_provider_policy(
    raw: Mapping[str, Any] | None,
    *,
    provider_id: str,
    processing_region: str,
    provider_retention_days: int,
) -> dict[str, Any]:
    """Validate and normalize the deployment-owned provider policy."""

    return _validated_provider_policy(
        raw,
        provider_id=provider_id,
        processing_region=processing_region,
        provider_retention_days=provider_retention_days,
    )


def validated_subject_policy(
    raw: Mapping[str, Any] | None,
    *,
    likely_minor: bool,
    guardian_or_school_policy: str,
) -> dict[str, Any]:
    """Validate the server-owned age/guardian processing policy."""

    return _validated_subject_policy(
        raw,
        likely_minor=likely_minor,
        guardian_or_school_policy=guardian_or_school_policy,
    )


def consent_policy_sha256(policy: Mapping[str, Any]) -> str:
    """Return the canonical public binding hash for a validated policy."""

    return _sha(dict(policy))


class RemoteConsentStore:
    """Atomic, hash-chained local consent authority.

    The file is a compact projection plus its immutable event chain.  Every
    mutation rereads under the inter-process lock, validates all hashes and
    signatures, then uses fsync + replace + directory fsync.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        signing_secret: bytes,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = Path(path).resolve()
        self.signing_secret = _validate_secret(signing_secret)
        self.clock = clock
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.parent.is_symlink():
            raise ConsentError("consent store directory must not be a symlink")
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError as exc:
            raise ConsentError("cannot secure consent store directory") from exc
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._ensure_lock_file()

    def _ensure_lock_file(self) -> None:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._lock_path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
        except OSError as exc:
            raise ConsentError("cannot secure consent store lock") from exc

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._lock:
            descriptor = os.open(self._lock_path, os.O_RDWR)
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _empty(self) -> dict[str, Any]:
        return {
            "schema": CONSENT_STORE_SCHEMA,
            "revision": 0,
            "head_sha256": "0" * 64,
            "events": [],
            "receipts": {},
        }

    def _signature(self, receipt_hash: str) -> str:
        return hmac.new(
            self.signing_secret, receipt_hash.encode("ascii"), hashlib.sha256
        ).hexdigest()

    def _validated_receipt(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ConsentError("consent receipt is invalid")
        receipt = deepcopy(dict(raw))
        if receipt.get("schema") != CONSENT_RECEIPT_SCHEMA:
            raise ConsentError("consent receipt schema is invalid")
        receipt_hash = receipt.pop("receipt_sha256", None)
        signature = receipt.pop("signature", None)
        if (
            not isinstance(receipt_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt_hash)
            or not isinstance(signature, str)
            or not re.fullmatch(r"[0-9a-f]{64}", signature)
            or not hmac.compare_digest(_sha(receipt), receipt_hash)
            or not hmac.compare_digest(self._signature(receipt_hash), signature)
        ):
            raise ConsentError("consent receipt seal is invalid")
        if receipt.get("policy_version") != CONSENT_POLICY_VERSION:
            raise ConsentError("consent policy version is unsupported")
        if receipt.get("status") not in {"active", "revoked"}:
            raise ConsentError("consent receipt status is invalid")
        for field in ("consent_id", "subject_id"):
            if not isinstance(receipt.get(field), str) or not _OPAQUE.fullmatch(
                receipt[field]
            ):
                raise ConsentError(f"consent receipt {field} is invalid")
        if receipt.get("purpose") not in _PURPOSES:
            raise ConsentError("consent purpose is invalid")
        if not isinstance(receipt.get("provider_id"), str) or not _PROVIDER.fullmatch(
            receipt["provider_id"]
        ):
            raise ConsentError("consent provider is invalid")
        if not isinstance(
            receipt.get("processing_region"), str
        ) or not _REGION.fullmatch(receipt["processing_region"]):
            raise ConsentError("consent processing region is invalid")
        categories = receipt.get("data_categories")
        if (
            not isinstance(categories, list)
            or not categories
            or categories != sorted(set(categories))
            or any(item not in _DATA_CATEGORIES for item in categories)
        ):
            raise ConsentError("consent data categories are invalid")
        if receipt.get("guardian_or_school_policy") not in _GUARDIAN_POLICIES:
            raise ConsentError("consent guardian policy is invalid")
        granted = _parse_time(receipt.get("granted_at_utc"))
        expires = _parse_time(receipt.get("expires_at_utc"))
        if expires <= granted or expires - granted > timedelta(days=365):
            raise ConsentError("consent validity interval is invalid")
        retention_days = receipt.get("provider_retention_days")
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, int)
            or not 0 <= retention_days <= 365
        ):
            raise ConsentError("consent provider retention is invalid")
        provider_policy = receipt.get("provider_policy")
        subject_policy = receipt.get("subject_policy")
        # Legacy signed receipts remain readable for revocation/audit, but a
        # production effect gate can (and does) reject their missing policy
        # hashes as stale. New grants always carry both sealed policies.
        if provider_policy is not None or subject_policy is not None:
            if not isinstance(provider_policy, Mapping) or not isinstance(
                subject_policy, Mapping
            ):
                raise ConsentError("consent policy bindings are invalid")
            validated_provider = _validated_provider_policy(
                provider_policy,
                provider_id=str(receipt["provider_id"]),
                processing_region=str(receipt["processing_region"]),
                provider_retention_days=retention_days,
            )
            validated_subject = _validated_subject_policy(
                subject_policy,
                likely_minor=bool(subject_policy.get("likely_minor")),
                guardian_or_school_policy=str(receipt["guardian_or_school_policy"]),
            )
            if receipt.get("provider_policy_sha256") != _sha(
                validated_provider
            ) or receipt.get("subject_policy_sha256") != _sha(validated_subject):
                raise ConsentError("consent policy binding hash is invalid")
        if receipt["status"] == "active" and receipt.get("revoked_at_utc") is not None:
            raise ConsentError("active consent cannot have a revocation time")
        if receipt["status"] == "revoked":
            _parse_time(receipt.get("revoked_at_utc"))
        return {**receipt, "receipt_sha256": receipt_hash, "signature": signature}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        if self.path.is_symlink():
            raise ConsentError("consent store must not be a symlink")
        metadata = self.path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise ConsentError("consent store ownership is invalid")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ConsentError("consent store permissions are too broad")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConsentError("consent store cannot be decoded") from exc
        if not isinstance(raw, Mapping) or raw.get("schema") != CONSENT_STORE_SCHEMA:
            raise ConsentError("consent store schema is invalid")
        events = raw.get("events")
        receipts = raw.get("receipts")
        if (
            not isinstance(events, list)
            or len(events) > MAX_CONSENT_EVENTS
            or not isinstance(receipts, Mapping)
            or raw.get("revision") != len(events)
        ):
            raise ConsentError("consent store projection is invalid")
        head = "0" * 64
        projected: dict[str, dict[str, Any]] = {}
        for sequence, event_raw in enumerate(events, 1):
            if not isinstance(event_raw, Mapping):
                raise ConsentError("consent event is invalid")
            event = dict(event_raw)
            event_hash = event.pop("event_sha256", None)
            if (
                event.get("sequence") != sequence
                or event.get("previous_sha256") != head
                or event.get("type") not in {"consent.granted", "consent.revoked"}
                or not isinstance(event_hash, str)
                or not hmac.compare_digest(_sha(event), event_hash)
            ):
                raise ConsentError("consent event chain is invalid")
            receipt = self._validated_receipt(event.get("receipt"))
            projected[receipt["consent_id"]] = receipt
            head = event_hash
        if raw.get("head_sha256") != head:
            raise ConsentError("consent store head is invalid")
        validated_receipts = {
            str(key): self._validated_receipt(value) for key, value in receipts.items()
        }
        if validated_receipts != projected:
            raise ConsentError("consent store projection diverges from its events")
        return {
            "schema": CONSENT_STORE_SCHEMA,
            "revision": len(events),
            "head_sha256": head,
            "events": deepcopy(events),
            "receipts": validated_receipts,
        }

    def _write(self, document: Mapping[str, Any]) -> None:
        encoded = _canonical(document)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".consent-", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            raise ConsentError("consent store atomic write failed") from exc
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _append(
        self, document: dict[str, Any], event_type: str, receipt: Mapping[str, Any]
    ) -> None:
        if len(document["events"]) >= MAX_CONSENT_EVENTS:
            raise ConsentError("consent store capacity is exhausted")
        event = {
            "sequence": len(document["events"]) + 1,
            "type": event_type,
            "previous_sha256": document["head_sha256"],
            "receipt": deepcopy(dict(receipt)),
        }
        event_hash = _sha(event)
        document["events"].append({**event, "event_sha256": event_hash})
        document["head_sha256"] = event_hash
        document["revision"] = len(document["events"])
        document["receipts"][receipt["consent_id"]] = deepcopy(dict(receipt))
        self._write(document)

    def grant(
        self,
        *,
        subject_id: str,
        purpose: str,
        provider_id: str,
        processing_region: str,
        data_categories: Sequence[str],
        provider_retention_days: int,
        validity_days: int = 30,
        likely_minor: bool = False,
        guardian_or_school_policy: str = "not_required",
        provider_policy: Mapping[str, Any] | None = None,
        subject_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not _OPAQUE.fullmatch(str(subject_id)):
            raise ConsentError("consent subject is invalid")
        if purpose not in _PURPOSES:
            raise ConsentError("consent purpose is invalid")
        if not _PROVIDER.fullmatch(str(provider_id)):
            raise ConsentError("consent provider is invalid")
        if not _REGION.fullmatch(str(processing_region)):
            raise ConsentError("consent processing region is invalid")
        categories = sorted(set(str(item) for item in data_categories))
        if not categories or any(item not in _DATA_CATEGORIES for item in categories):
            raise ConsentError("consent data categories are invalid")
        if (
            isinstance(provider_retention_days, bool)
            or not isinstance(provider_retention_days, int)
            or not 0 <= provider_retention_days <= 365
        ):
            raise ConsentError("provider_retention_days is invalid")
        if (
            isinstance(validity_days, bool)
            or not isinstance(validity_days, int)
            or not 1 <= validity_days <= 365
        ):
            raise ConsentError("validity_days is invalid")
        if guardian_or_school_policy not in _GUARDIAN_POLICIES:
            raise ConsentError("guardian_or_school_policy is invalid")
        if likely_minor and guardian_or_school_policy == "not_required":
            raise ConsentError(
                "minor remote processing requires guardian or school policy"
            )
        validated_provider_policy = _validated_provider_policy(
            provider_policy,
            provider_id=str(provider_id),
            processing_region=str(processing_region),
            provider_retention_days=provider_retention_days,
        )
        validated_subject_policy = _validated_subject_policy(
            subject_policy,
            likely_minor=likely_minor,
            guardian_or_school_policy=guardian_or_school_policy,
        )
        if validated_subject_policy["remote_processing_eligible"] is not True:
            raise ConsentError("subject policy does not authorize remote processing")
        now = self.clock()
        consent_id = "consent_" + secrets.token_urlsafe(18)
        unsigned = {
            "schema": CONSENT_RECEIPT_SCHEMA,
            "consent_id": consent_id,
            "subject_id": subject_id,
            "purpose": purpose,
            "provider_id": provider_id,
            "processing_region": processing_region,
            "data_categories": categories,
            "provider_retention_days": provider_retention_days,
            "policy_version": CONSENT_POLICY_VERSION,
            "guardian_or_school_policy": guardian_or_school_policy,
            "provider_policy": validated_provider_policy,
            "provider_policy_sha256": _sha(validated_provider_policy),
            "subject_policy": validated_subject_policy,
            "subject_policy_sha256": _sha(validated_subject_policy),
            "status": "active",
            "granted_at_utc": _iso(now),
            "expires_at_utc": _iso(now + timedelta(days=validity_days)),
            "revoked_at_utc": None,
            "revocation_reason_code": None,
        }
        receipt_hash = _sha(unsigned)
        receipt = {
            **unsigned,
            "receipt_sha256": receipt_hash,
            "signature": self._signature(receipt_hash),
        }
        self._validated_receipt(receipt)
        with self._exclusive():
            document = self._read()
            self._append(document, "consent.granted", receipt)
        return deepcopy(receipt)

    def verify(
        self,
        consent_id: str,
        *,
        subject_id: str,
        purpose: str,
        provider_id: str,
        required_data_categories: Sequence[str],
        provider_policy: Mapping[str, Any] | None = None,
        subject_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._exclusive():
            receipt = self._read()["receipts"].get(str(consent_id))
        if receipt is None:
            raise ConsentError("consent receipt is not available")
        receipt = self._validated_receipt(receipt)
        if (
            receipt["subject_id"] != subject_id
            or receipt["purpose"] != purpose
            or receipt["provider_id"] != provider_id
        ):
            raise ConsentError("consent receipt does not authorize this effect")
        required = set(str(item) for item in required_data_categories)
        if not required or not required.issubset(set(receipt["data_categories"])):
            raise ConsentError("consent receipt does not cover the requested data")
        if receipt["status"] != "active":
            raise ConsentError("consent receipt has been revoked")
        if self.clock().astimezone(timezone.utc) >= _parse_time(
            receipt["expires_at_utc"]
        ):
            raise ConsentError("consent receipt has expired")
        if (provider_policy is None) is not (subject_policy is None):
            raise ConsentError("current consent policies must be supplied together")
        if provider_policy is not None and subject_policy is not None:
            validated_provider = _validated_provider_policy(
                provider_policy,
                provider_id=str(receipt["provider_id"]),
                processing_region=str(receipt["processing_region"]),
                provider_retention_days=int(receipt["provider_retention_days"]),
            )
            validated_subject = _validated_subject_policy(
                subject_policy,
                likely_minor=bool(subject_policy.get("likely_minor")),
                guardian_or_school_policy=str(
                    subject_policy.get("guardian_or_school_policy", "")
                ),
            )
            if receipt.get("provider_policy_sha256") != _sha(
                validated_provider
            ) or receipt.get("subject_policy_sha256") != _sha(validated_subject):
                raise ConsentError("consent receipt policy is stale")
        return deepcopy(receipt)

    def revoke(
        self, consent_id: str, *, subject_id: str, reason_code: str
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", str(reason_code)):
            raise ConsentError("consent revocation reason is invalid")
        with self._exclusive():
            document = self._read()
            previous = document["receipts"].get(str(consent_id))
            if previous is None or previous["subject_id"] != subject_id:
                raise ConsentError("consent receipt is not available")
            if previous["status"] == "revoked":
                return deepcopy(previous)
            unsigned = {
                key: deepcopy(value)
                for key, value in previous.items()
                if key not in {"receipt_sha256", "signature"}
            }
            unsigned.update(
                {
                    "status": "revoked",
                    "revoked_at_utc": _iso(self.clock()),
                    "revocation_reason_code": reason_code,
                }
            )
            receipt_hash = _sha(unsigned)
            receipt = {
                **unsigned,
                "receipt_sha256": receipt_hash,
                "signature": self._signature(receipt_hash),
            }
            self._validated_receipt(receipt)
            self._append(document, "consent.revoked", receipt)
            return deepcopy(receipt)

    def list_for_subject(self, subject_id: str) -> list[dict[str, Any]]:
        with self._exclusive():
            receipts = self._read()["receipts"].values()
        return sorted(
            (
                deepcopy(receipt)
                for receipt in receipts
                if receipt["subject_id"] == subject_id
            ),
            key=lambda item: (item["granted_at_utc"], item["consent_id"]),
            reverse=True,
        )


__all__ = [
    "CONSENT_POLICY_VERSION",
    "CONSENT_RECEIPT_SCHEMA",
    "CONSENT_STORE_SCHEMA",
    "ConsentError",
    "RemoteConsentStore",
    "consent_policy_sha256",
    "validated_provider_policy",
    "validated_subject_policy",
]
