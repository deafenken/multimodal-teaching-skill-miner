"""Durable teacher review and curriculum-grading authority ledger.

The immutable syllabus/version stores answer which syllabus revision is
published.  This ledger answers a separate question: whether an authenticated
teacher reviewed an exact curriculum/measurement specification for that exact
published revision and whether that authority is still active.

Every mutation is scope-bound, CAS fenced, idempotent, hash chained and
atomically replaced.  Runtime callers revalidate both the gateway receipt and
the Ed25519 curriculum seal on every read; copying a ledger into another
tenant/owner worker therefore fails closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

try:  # pragma: no cover - production deployment is POSIX.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from .io_utils import ensure_private_directory
from .teacher_agent_authority import (
    TeacherAuthorityError,
    validate_teacher_authority_verification_receipt,
)
from .teacher_agent_curriculum import (
    CurriculumBlueprintError,
    validate_curriculum_blueprint,
    verify_teacher_curriculum_runtime_authority,
)


CURRICULUM_AUTHORITY_STORE_SCHEMA = (
    "teaching_skill_miner.curriculum_authority_store.v1"
)
CURRICULUM_AUTHORITY_EVENT_SCHEMA = (
    "teaching_skill_miner.curriculum_authority_event.v1"
)
CURRICULUM_AUTHORITY_PROJECTION_SCHEMA = (
    "teaching_skill_miner.curriculum_authority_projection.v1"
)
CURRICULUM_RUNTIME_AUTHORITY_SCHEMA = (
    "teaching_skill_miner.curriculum_runtime_authority.v1"
)

_SCOPE = re.compile(r"^scope_[0-9a-f]{48}$")
_FAMILY = re.compile(r"^syf_[0-9a-f]{24}$")
_REVISION = re.compile(r"^syr_[0-9a-f]{24}$")
_SYLLABUS = re.compile(r"^syl_[0-9a-f]{24}$")
_CURRICULUM = re.compile(r"^cur_[0-9a-f]{24}$")
_REVIEW = re.compile(r"^currev_[0-9a-f]{24}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_REASON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,79}$")
_MAX_EVENTS = 8_192
_MAX_BYTES = 64 * 1024 * 1024

_EVENT_PAYLOAD_KEYS = {
    "review.recorded": {
        "family_id",
        "authority_version",
        "published_revision_id",
        "published_syllabus_id",
        "published_syllabus_sha256",
        "review_id",
        "teacher_spec",
        "teacher_spec_sha256",
        "gateway_authority_receipt",
        "gateway_authority_receipt_sha256",
    },
    "authority.sealed": {
        "family_id",
        "authority_version",
        "published_revision_id",
        "published_syllabus_id",
        "published_syllabus_sha256",
        "review_id",
        "teacher_spec_sha256",
        "curriculum_id",
        "curriculum_blueprint",
        "gateway_authority_receipt",
        "gateway_authority_receipt_sha256",
    },
    "authority.revoked": {
        "family_id",
        "authority_version",
        "published_revision_id",
        "published_syllabus_id",
        "published_syllabus_sha256",
        "curriculum_id",
        "reason_code",
        "gateway_authority_receipt",
        "gateway_authority_receipt_sha256",
    },
}

_EVENT_ROUTE = {
    "review.recorded": "api/curriculum/review",
    "authority.sealed": "api/curriculum/seal",
    "authority.revoked": "api/curriculum/revoke",
}


class CurriculumAuthorityStoreError(RuntimeError):
    """Raised when durable curriculum authority cannot be proven safe."""


class CurriculumAuthorityConflictError(CurriculumAuthorityStoreError):
    """Raised for CAS or idempotency conflicts."""


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
        raise CurriculumAuthorityStoreError(
            "curriculum authority state must be canonical JSON"
        ) from exc


def _hash(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _utc(value: str | None = None) -> str:
    result = value or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not isinstance(result, str) or _UTC.fullmatch(result) is None:
        raise CurriculumAuthorityStoreError(
            "curriculum authority timestamp must use UTC second precision"
        )
    try:
        datetime.strptime(result, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise CurriculumAuthorityStoreError(
            "curriculum authority timestamp is not real"
        ) from exc
    return result


def _binding(
    *,
    family_id: str,
    published_revision_id: str,
    published_syllabus_id: str,
    published_syllabus_sha256: str,
) -> dict[str, str]:
    if (
        _FAMILY.fullmatch(family_id) is None
        or _REVISION.fullmatch(published_revision_id) is None
        or _SYLLABUS.fullmatch(published_syllabus_id) is None
        or _DIGEST.fullmatch(published_syllabus_sha256) is None
    ):
        raise CurriculumAuthorityStoreError(
            "published syllabus authority binding is invalid"
        )
    return {
        "family_id": family_id,
        "published_revision_id": published_revision_id,
        "published_syllabus_id": published_syllabus_id,
        "published_syllabus_sha256": published_syllabus_sha256,
    }


def _event_hash(event: Mapping[str, Any]) -> str:
    material = deepcopy(dict(event))
    material.pop("event_sha256", None)
    return _hash(material)


def _empty_state(scope_id: str) -> dict[str, Any]:
    return {
        "schema": CURRICULUM_AUTHORITY_STORE_SCHEMA,
        "version": 1,
        "scope_id": scope_id,
        "event_sequence": 0,
        "events": [],
        "erasure_tombstones": {},
    }


def _tombstone_key(scope_id: str, family_id: str) -> str:
    return sha256(f"curriculum-family-v1\0{scope_id}\0{family_id}".encode()).hexdigest()


def _gateway_receipt(
    value: Any,
    *,
    expected_scope_ids: frozenset[str],
    expected_path: str,
    expected_request_sha256: str,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CurriculumAuthorityStoreError(
            "curriculum authority gateway receipt is missing"
        )
    try:
        structural = validate_teacher_authority_verification_receipt(value)
        checked = validator(structural)
        checked = validate_teacher_authority_verification_receipt(checked)
    except (TeacherAuthorityError, TypeError, ValueError) as exc:
        raise CurriculumAuthorityStoreError(
            "curriculum authority gateway receipt is invalid"
        ) from exc
    if (
        checked.get("scope_id") not in expected_scope_ids
        or checked.get("method") != "POST"
        or checked.get("path") != expected_path
        or checked.get("body_sha256") != expected_request_sha256
    ):
        raise CurriculumAuthorityStoreError(
            "curriculum authority gateway receipt binding is invalid"
        )
    return deepcopy(dict(checked))


class TeachingCurriculumAuthorityStore:
    """Cross-process-safe review/seal/revocation ledger."""

    def __init__(
        self,
        path: str | Path,
        *,
        scope_id: str,
        gateway_scope_ids: Sequence[str] | None = None,
        trusted_teacher_public_keys: Mapping[str, bytes | str]
        | Callable[[], Mapping[str, bytes | str]],
        known_teacher_public_keys: Mapping[str, bytes | str]
        | Callable[[], Mapping[str, bytes | str]]
        | None = None,
        gateway_receipt_validator: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ],
    ) -> None:
        if _SCOPE.fullmatch(scope_id) is None:
            raise CurriculumAuthorityStoreError(
                "curriculum authority scope is invalid"
            )
        candidate = Path(path).expanduser()
        if candidate.exists() and candidate.is_symlink():
            raise CurriculumAuthorityStoreError(
                "curriculum authority store cannot be a symlink"
            )
        self.root = ensure_private_directory(candidate.parent).resolve()
        self.path = self.root / candidate.name
        self.lock_path = self.root / f".{candidate.name}.lock"
        self.scope_id = scope_id
        raw_gateway_scope_ids = (
            [scope_id] if gateway_scope_ids is None else list(gateway_scope_ids)
        )
        if (
            not raw_gateway_scope_ids
            or any(
                not isinstance(item, str) or _SCOPE.fullmatch(item) is None
                for item in raw_gateway_scope_ids
            )
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority gateway scopes are invalid"
        )
        self.gateway_scope_ids = frozenset(raw_gateway_scope_ids)
        self._trusted_teacher_public_keys = trusted_teacher_public_keys
        self._known_teacher_public_keys = (
            trusted_teacher_public_keys
            if known_teacher_public_keys is None
            else known_teacher_public_keys
        )
        if not self._trusted_keys():
            raise CurriculumAuthorityStoreError(
                "curriculum authority trust registry is empty"
            )
        if not self._known_keys():
            raise CurriculumAuthorityStoreError(
                "curriculum authority audit registry is empty"
            )
        self._gateway_receipt_validator = gateway_receipt_validator
        self._lock = threading.RLock()

    def _trusted_keys(self) -> dict[str, bytes | str]:
        raw = (
            self._trusted_teacher_public_keys()
            if callable(self._trusted_teacher_public_keys)
            else self._trusted_teacher_public_keys
        )
        if not isinstance(raw, Mapping):
            raise CurriculumAuthorityStoreError(
                "curriculum authority trust registry is invalid"
            )
        return dict(raw)

    def _known_keys(self) -> dict[str, bytes | str]:
        raw = (
            self._known_teacher_public_keys()
            if callable(self._known_teacher_public_keys)
            else self._known_teacher_public_keys
        )
        if not isinstance(raw, Mapping):
            raise CurriculumAuthorityStoreError(
                "curriculum authority audit registry is invalid"
            )
        return dict(raw)

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._lock:
            descriptor = os.open(
                self.lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _validate_event(
        self,
        event: Any,
        *,
        sequence: int,
        previous_hash: str | None,
    ) -> dict[str, Any]:
        keys = {
            "schema",
            "sequence",
            "event_id",
            "event_type",
            "occurred_at_utc",
            "idempotency_key",
            "request_sha256",
            "payload",
            "previous_event_sha256",
            "event_sha256",
        }
        if not isinstance(event, Mapping) or set(event) != keys:
            raise CurriculumAuthorityStoreError(
                "curriculum authority event shape is invalid"
            )
        event_type = event.get("event_type")
        if (
            event.get("schema") != CURRICULUM_AUTHORITY_EVENT_SCHEMA
            or event.get("sequence") != sequence
            or event_type not in _EVENT_PAYLOAD_KEYS
            or not isinstance(event.get("event_id"), str)
            or re.fullmatch(r"curae_[0-9a-f]{24}", str(event["event_id"])) is None
            or not isinstance(event.get("idempotency_key"), str)
            or _IDEMPOTENCY.fullmatch(str(event["idempotency_key"])) is None
            or not isinstance(event.get("request_sha256"), str)
            or _DIGEST.fullmatch(str(event["request_sha256"])) is None
            or event.get("previous_event_sha256") != previous_hash
            or event.get("event_sha256") != _event_hash(event)
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority event integrity is invalid"
            )
        _utc(str(event.get("occurred_at_utc", "")))
        payload = event.get("payload")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _EVENT_PAYLOAD_KEYS[str(event_type)]
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority event payload is invalid"
            )
        bound = _binding(
            family_id=str(payload.get("family_id", "")),
            published_revision_id=str(payload.get("published_revision_id", "")),
            published_syllabus_id=str(payload.get("published_syllabus_id", "")),
            published_syllabus_sha256=str(
                payload.get("published_syllabus_sha256", "")
            ),
        )
        authority_version = payload.get("authority_version")
        if (
            isinstance(authority_version, bool)
            or not isinstance(authority_version, int)
            or authority_version < 1
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority event version is invalid"
            )
        receipt = _gateway_receipt(
            payload.get("gateway_authority_receipt"),
            expected_scope_ids=self.gateway_scope_ids,
            expected_path=_EVENT_ROUTE[str(event_type)],
            expected_request_sha256=str(event["request_sha256"]),
            validator=self._gateway_receipt_validator,
        )
        if payload.get("gateway_authority_receipt_sha256") != receipt.get(
            "receipt_sha256"
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority receipt hash is invalid"
            )
        if event_type == "review.recorded":
            spec = payload.get("teacher_spec")
            if (
                not isinstance(spec, Mapping)
                or _DIGEST.fullmatch(str(payload.get("teacher_spec_sha256", "")))
                is None
                or payload.get("teacher_spec_sha256") != _hash(spec)
                or _REVIEW.fullmatch(str(payload.get("review_id", ""))) is None
                or payload.get("review_id")
                != "currev_"
                + _hash(
                    {
                        **bound,
                        "teacher_spec_sha256": payload["teacher_spec_sha256"],
                    }
                )[:24]
            ):
                raise CurriculumAuthorityStoreError(
                    "curriculum review content binding is invalid"
                )
        elif event_type == "authority.sealed":
            blueprint = payload.get("curriculum_blueprint")
            if not isinstance(blueprint, Mapping):
                raise CurriculumAuthorityStoreError(
                    "sealed curriculum blueprint is missing"
                )
            try:
                validate_curriculum_blueprint(
                    blueprint,
                    # Historic signatures remain auditable after an explicit
                    # key revocation. Runtime authority below still accepts
                    # only the current non-revoked trust registry.
                    trusted_teacher_public_keys=self._known_keys(),
                )
            except CurriculumBlueprintError as exc:
                raise CurriculumAuthorityStoreError(
                    "sealed curriculum blueprint is invalid"
                ) from exc
            if (
                payload.get("curriculum_id") != blueprint.get("curriculum_id")
                or _CURRICULUM.fullmatch(str(payload.get("curriculum_id", "")))
                is None
                or _REVIEW.fullmatch(str(payload.get("review_id", ""))) is None
                or _DIGEST.fullmatch(str(payload.get("teacher_spec_sha256", "")))
                is None
                or blueprint.get("origin", {}).get("source_sha256")
                != payload.get("teacher_spec_sha256")
            ):
                raise CurriculumAuthorityStoreError(
                    "sealed curriculum identity binding is invalid"
                )
        else:
            if (
                _CURRICULUM.fullmatch(str(payload.get("curriculum_id", "")))
                is None
                or _REASON.fullmatch(str(payload.get("reason_code", ""))) is None
            ):
                raise CurriculumAuthorityStoreError(
                    "curriculum revocation content is invalid"
                )
        return deepcopy(dict(event))

    def _project(
        self, state: Mapping[str, Any], *, through_sequence: int | None = None
    ) -> dict[str, dict[str, Any]]:
        families: dict[str, dict[str, Any]] = {}
        events = state["events"]
        if through_sequence is not None:
            events = events[:through_sequence]
        for event in events:
            payload = event["payload"]
            family_id = str(payload["family_id"])
            current = families.get(family_id)
            expected_version = 1 if current is None else int(current["version"]) + 1
            if payload["authority_version"] != expected_version:
                raise CurriculumAuthorityStoreError(
                    "curriculum authority family version is not contiguous"
                )
            binding = {
                key: deepcopy(payload[key])
                for key in (
                    "published_revision_id",
                    "published_syllabus_id",
                    "published_syllabus_sha256",
                )
            }
            event_type = event["event_type"]
            if event_type == "review.recorded":
                next_projection = {
                    "schema": CURRICULUM_AUTHORITY_PROJECTION_SCHEMA,
                    "family_id": family_id,
                    "version": expected_version,
                    "status": "reviewed",
                    **binding,
                    "review": {
                        "review_id": payload["review_id"],
                        "teacher_spec": deepcopy(payload["teacher_spec"]),
                        "teacher_spec_sha256": payload["teacher_spec_sha256"],
                        "gateway_authority_receipt": deepcopy(
                            payload["gateway_authority_receipt"]
                        ),
                        "gateway_authority_receipt_sha256": payload[
                            "gateway_authority_receipt_sha256"
                        ],
                    },
                    "seal": None,
                    "revocation": None,
                    "event_ids": [str(event["event_id"])],
                }
            elif event_type == "authority.sealed":
                if (
                    current is None
                    or current["status"] != "reviewed"
                    or current["review"]["review_id"] != payload["review_id"]
                    or current["review"]["teacher_spec_sha256"]
                    != payload["teacher_spec_sha256"]
                    or any(current[key] != value for key, value in binding.items())
                ):
                    raise CurriculumAuthorityStoreError(
                        "curriculum seal does not follow its exact review"
                    )
                next_projection = {
                    **deepcopy(current),
                    "version": expected_version,
                    "status": "sealed",
                    "seal": {
                        "curriculum_id": payload["curriculum_id"],
                        "curriculum_blueprint": deepcopy(
                            payload["curriculum_blueprint"]
                        ),
                        "gateway_authority_receipt": deepcopy(
                            payload["gateway_authority_receipt"]
                        ),
                        "gateway_authority_receipt_sha256": payload[
                            "gateway_authority_receipt_sha256"
                        ],
                    },
                    "event_ids": [
                        *current["event_ids"],
                        str(event["event_id"]),
                    ],
                }
            else:
                if (
                    current is None
                    or current["status"] != "sealed"
                    or current["seal"]["curriculum_id"]
                    != payload["curriculum_id"]
                    or any(current[key] != value for key, value in binding.items())
                ):
                    raise CurriculumAuthorityStoreError(
                        "curriculum revocation does not target an active seal"
                    )
                next_projection = {
                    **deepcopy(current),
                    "version": expected_version,
                    "status": "revoked",
                    "revocation": {
                        "reason_code": payload["reason_code"],
                        "gateway_authority_receipt": deepcopy(
                            payload["gateway_authority_receipt"]
                        ),
                        "gateway_authority_receipt_sha256": payload[
                            "gateway_authority_receipt_sha256"
                        ],
                    },
                    "event_ids": [
                        *current["event_ids"],
                        str(event["event_id"]),
                    ],
                }
            families[family_id] = next_projection
        return families

    def _validate_state(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "version",
            "scope_id",
            "event_sequence",
            "events",
            "erasure_tombstones",
        }:
            raise CurriculumAuthorityStoreError(
                "curriculum authority store shape is invalid"
            )
        if (
            value.get("schema") != CURRICULUM_AUTHORITY_STORE_SCHEMA
            or value.get("version") != 1
            or value.get("scope_id") != self.scope_id
            or not isinstance(value.get("events"), list)
            or value.get("event_sequence") != len(value["events"])
            or len(value["events"]) > _MAX_EVENTS
            or not isinstance(value.get("erasure_tombstones"), Mapping)
            or len(value["erasure_tombstones"]) > _MAX_EVENTS
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority store boundary is invalid"
            )
        previous: str | None = None
        idempotency: dict[str, tuple[str, str]] = {}
        for sequence, raw in enumerate(value["events"], start=1):
            event = self._validate_event(
                raw, sequence=sequence, previous_hash=previous
            )
            key = str(event["idempotency_key"])
            identity = (str(event["event_type"]), str(event["request_sha256"]))
            if key in idempotency and idempotency[key] != identity:
                raise CurriculumAuthorityStoreError(
                    "curriculum authority idempotency history conflicts"
                )
            idempotency[key] = identity
            previous = str(event["event_sha256"])
        tombstones = value["erasure_tombstones"]
        for digest, tombstone in tombstones.items():
            if (
                not isinstance(digest, str)
                or _DIGEST.fullmatch(digest) is None
                or not isinstance(tombstone, Mapping)
                or set(tombstone) != {"purged_at_utc", "generation"}
                or isinstance(tombstone.get("generation"), bool)
                or not isinstance(tombstone.get("generation"), int)
                or int(tombstone["generation"]) < 1
            ):
                raise CurriculumAuthorityStoreError(
                    "curriculum authority erasure tombstone is invalid"
                )
            _utc(str(tombstone.get("purged_at_utc", "")))
        families = self._project(value)
        if any(
            _tombstone_key(self.scope_id, family_id) in tombstones
            for family_id in families
        ):
            raise CurriculumAuthorityStoreError(
                "erased curriculum authority was resurrected"
            )
        if len(_canonical(value)) > _MAX_BYTES:
            raise CurriculumAuthorityStoreError(
                "curriculum authority store exceeds its byte limit"
            )
        return deepcopy(dict(value))

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_state(self.scope_id)
        if self.path.is_symlink() or not self.path.is_file():
            raise CurriculumAuthorityStoreError(
                "curriculum authority store path is unsafe"
            )
        try:
            raw = self.path.read_bytes()
            if len(raw) > _MAX_BYTES:
                raise CurriculumAuthorityStoreError(
                    "curriculum authority store exceeds its byte limit"
                )
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CurriculumAuthorityStoreError(
                "curriculum authority store cannot be read"
            ) from exc
        return self._validate_state(value)

    def _write(self, state: Mapping[str, Any]) -> None:
        candidate = self._validate_state(state)
        payload = _canonical(candidate) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                handle.write(payload)
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _append(
        self,
        state: dict[str, Any],
        *,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
        request_sha256: str,
        occurred_at_utc: str,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        if (
            _IDEMPOTENCY.fullmatch(idempotency_key) is None
            or _DIGEST.fullmatch(request_sha256) is None
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority mutation identity is invalid"
            )
        for event in state["events"]:
            if event["idempotency_key"] != idempotency_key:
                continue
            if (
                event["event_type"] != event_type
                or event["request_sha256"] != request_sha256
            ):
                raise CurriculumAuthorityConflictError(
                    "curriculum authority idempotency key was reused"
                )
            return state, deepcopy(dict(event)), False
        if len(state["events"]) >= _MAX_EVENTS:
            raise CurriculumAuthorityStoreError(
                "curriculum authority event capacity is exhausted"
            )
        sequence = len(state["events"]) + 1
        previous = state["events"][-1]["event_sha256"] if state["events"] else None
        event: dict[str, Any] = {
            "schema": CURRICULUM_AUTHORITY_EVENT_SCHEMA,
            "sequence": sequence,
            "event_id": "curae_"
            + _hash(
                {
                    "sequence": sequence,
                    "event_type": event_type,
                    "request_sha256": request_sha256,
                    "previous": previous,
                }
            )[:24],
            "event_type": event_type,
            "occurred_at_utc": occurred_at_utc,
            "idempotency_key": idempotency_key,
            "request_sha256": request_sha256,
            "payload": deepcopy(dict(payload)),
            "previous_event_sha256": previous,
        }
        event["event_sha256"] = _event_hash(event)
        state["events"].append(event)
        state["event_sequence"] = sequence
        self._write(state)
        return state, deepcopy(event), True

    @staticmethod
    def _receipt_request_sha(receipt: Mapping[str, Any], expected_path: str) -> str:
        checked = validate_teacher_authority_verification_receipt(receipt)
        if checked.get("path") != expected_path or checked.get("method") != "POST":
            raise CurriculumAuthorityStoreError(
                "curriculum authority receipt operation is invalid"
            )
        return str(checked["body_sha256"])

    def record_review(
        self,
        teacher_spec: Mapping[str, Any],
        *,
        family_id: str,
        published_revision_id: str,
        published_syllabus_id: str,
        published_syllabus_sha256: str,
        expected_version: int,
        idempotency_key: str,
        gateway_authority_receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        bound = _binding(
            family_id=family_id,
            published_revision_id=published_revision_id,
            published_syllabus_id=published_syllabus_id,
            published_syllabus_sha256=published_syllabus_sha256,
        )
        if not isinstance(teacher_spec, Mapping):
            raise CurriculumAuthorityStoreError(
                "teacher curriculum review must be an object"
            )
        spec = deepcopy(dict(teacher_spec))
        spec_sha = _hash(spec)
        review_id = "currev_" + _hash({**bound, "teacher_spec_sha256": spec_sha})[:24]
        request_sha = self._receipt_request_sha(
            gateway_authority_receipt, "api/curriculum/review"
        )
        with self._guard():
            state = self._read()
            replay = next(
                (
                    event
                    for event in state["events"]
                    if event["idempotency_key"] == idempotency_key
                ),
                None,
            )
            if replay is not None:
                if (
                    replay["event_type"] != "review.recorded"
                    or replay["request_sha256"] != request_sha
                ):
                    raise CurriculumAuthorityConflictError(
                        "curriculum authority idempotency key was reused"
                    )
                return self._projection_at(state, replay), False
            families = self._project(state)
            current = families.get(family_id)
            current_version = int(current["version"]) if current is not None else 0
            if current_version != expected_version:
                raise CurriculumAuthorityConflictError(
                    "curriculum authority version conflict"
                )
            if (
                current is not None
                and current["status"] == "sealed"
                and all(current[key] == value for key, value in bound.items() if key != "family_id")
            ):
                raise CurriculumAuthorityConflictError(
                    "active curriculum authority must be revoked before re-review"
                )
            if _tombstone_key(self.scope_id, family_id) in state["erasure_tombstones"]:
                raise CurriculumAuthorityStoreError(
                    "curriculum authority family was permanently erased"
                )
            checked_receipt = _gateway_receipt(
                gateway_authority_receipt,
                expected_scope_ids=self.gateway_scope_ids,
                expected_path="api/curriculum/review",
                expected_request_sha256=request_sha,
                validator=self._gateway_receipt_validator,
            )
            payload = {
                **bound,
                "authority_version": current_version + 1,
                "review_id": review_id,
                "teacher_spec": spec,
                "teacher_spec_sha256": spec_sha,
                "gateway_authority_receipt": checked_receipt,
                "gateway_authority_receipt_sha256": checked_receipt[
                    "receipt_sha256"
                ],
            }
            state, event, created = self._append(
                state,
                event_type="review.recorded",
                payload=payload,
                idempotency_key=idempotency_key,
                request_sha256=request_sha,
                occurred_at_utc=str(checked_receipt["issued_at"]),
            )
            return self._projection_at(state, event), created

    def seal_review(
        self,
        blueprint: Mapping[str, Any],
        *,
        family_id: str,
        published_revision_id: str,
        published_syllabus_id: str,
        published_syllabus_sha256: str,
        review_id: str,
        teacher_spec_sha256: str,
        expected_version: int,
        idempotency_key: str,
        gateway_authority_receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        bound = _binding(
            family_id=family_id,
            published_revision_id=published_revision_id,
            published_syllabus_id=published_syllabus_id,
            published_syllabus_sha256=published_syllabus_sha256,
        )
        try:
            validate_curriculum_blueprint(
                blueprint,
                trusted_teacher_public_keys=self._trusted_keys(),
            )
        except CurriculumBlueprintError as exc:
            raise CurriculumAuthorityStoreError(
                "sealed curriculum blueprint is invalid"
            ) from exc
        request_sha = self._receipt_request_sha(
            gateway_authority_receipt, "api/curriculum/seal"
        )
        with self._guard():
            state = self._read()
            replay = next(
                (
                    event
                    for event in state["events"]
                    if event["idempotency_key"] == idempotency_key
                ),
                None,
            )
            if replay is not None:
                if (
                    replay["event_type"] != "authority.sealed"
                    or replay["request_sha256"] != request_sha
                ):
                    raise CurriculumAuthorityConflictError(
                        "curriculum authority idempotency key was reused"
                    )
                return self._projection_at(state, replay), False
            current = self._project(state).get(family_id)
            if current is None or int(current["version"]) != expected_version:
                raise CurriculumAuthorityConflictError(
                    "curriculum authority version conflict"
                )
            if (
                current["status"] != "reviewed"
                or current["review"]["review_id"] != review_id
                or current["review"]["teacher_spec_sha256"]
                != teacher_spec_sha256
                or any(current[key] != value for key, value in bound.items() if key != "family_id")
            ):
                raise CurriculumAuthorityConflictError(
                    "curriculum seal does not match the active review"
                )
            if blueprint.get("origin", {}).get("source_sha256") != teacher_spec_sha256:
                raise CurriculumAuthorityStoreError(
                    "curriculum seal does not bind the reviewed specification"
                )
            checked_receipt = _gateway_receipt(
                gateway_authority_receipt,
                expected_scope_ids=self.gateway_scope_ids,
                expected_path="api/curriculum/seal",
                expected_request_sha256=request_sha,
                validator=self._gateway_receipt_validator,
            )
            payload = {
                **bound,
                "authority_version": expected_version + 1,
                "review_id": review_id,
                "teacher_spec_sha256": teacher_spec_sha256,
                "curriculum_id": str(blueprint["curriculum_id"]),
                "curriculum_blueprint": deepcopy(dict(blueprint)),
                "gateway_authority_receipt": checked_receipt,
                "gateway_authority_receipt_sha256": checked_receipt[
                    "receipt_sha256"
                ],
            }
            state, event, created = self._append(
                state,
                event_type="authority.sealed",
                payload=payload,
                idempotency_key=idempotency_key,
                request_sha256=request_sha,
                occurred_at_utc=str(checked_receipt["issued_at"]),
            )
            return self._projection_at(state, event), created

    def revoke(
        self,
        *,
        family_id: str,
        published_revision_id: str,
        published_syllabus_id: str,
        published_syllabus_sha256: str,
        curriculum_id: str,
        reason_code: str,
        expected_version: int,
        idempotency_key: str,
        gateway_authority_receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        bound = _binding(
            family_id=family_id,
            published_revision_id=published_revision_id,
            published_syllabus_id=published_syllabus_id,
            published_syllabus_sha256=published_syllabus_sha256,
        )
        if (
            _CURRICULUM.fullmatch(curriculum_id) is None
            or _REASON.fullmatch(reason_code) is None
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum revocation request is invalid"
            )
        request_sha = self._receipt_request_sha(
            gateway_authority_receipt, "api/curriculum/revoke"
        )
        with self._guard():
            state = self._read()
            replay = next(
                (
                    event
                    for event in state["events"]
                    if event["idempotency_key"] == idempotency_key
                ),
                None,
            )
            if replay is not None:
                if (
                    replay["event_type"] != "authority.revoked"
                    or replay["request_sha256"] != request_sha
                ):
                    raise CurriculumAuthorityConflictError(
                        "curriculum authority idempotency key was reused"
                    )
                return self._projection_at(state, replay), False
            current = self._project(state).get(family_id)
            if current is None or int(current["version"]) != expected_version:
                raise CurriculumAuthorityConflictError(
                    "curriculum authority version conflict"
                )
            if (
                current["status"] != "sealed"
                or current["seal"]["curriculum_id"] != curriculum_id
                or any(current[key] != value for key, value in bound.items() if key != "family_id")
            ):
                raise CurriculumAuthorityConflictError(
                    "curriculum revocation does not match the active seal"
                )
            checked_receipt = _gateway_receipt(
                gateway_authority_receipt,
                expected_scope_ids=self.gateway_scope_ids,
                expected_path="api/curriculum/revoke",
                expected_request_sha256=request_sha,
                validator=self._gateway_receipt_validator,
            )
            payload = {
                **bound,
                "authority_version": expected_version + 1,
                "curriculum_id": curriculum_id,
                "reason_code": reason_code,
                "gateway_authority_receipt": checked_receipt,
                "gateway_authority_receipt_sha256": checked_receipt[
                    "receipt_sha256"
                ],
            }
            state, event, created = self._append(
                state,
                event_type="authority.revoked",
                payload=payload,
                idempotency_key=idempotency_key,
                request_sha256=request_sha,
                occurred_at_utc=str(checked_receipt["issued_at"]),
            )
            return self._projection_at(state, event), created

    def _projection_at(
        self, state: Mapping[str, Any], event: Mapping[str, Any]
    ) -> dict[str, Any]:
        family_id = str(event["payload"]["family_id"])
        projection = self._project(
            state, through_sequence=int(event["sequence"])
        ).get(family_id)
        if projection is None:
            raise CurriculumAuthorityStoreError(
                "curriculum authority replay projection is unavailable"
            )
        return deepcopy(projection)

    def read_family(self, family_id: str) -> dict[str, Any] | None:
        if _FAMILY.fullmatch(family_id) is None:
            raise CurriculumAuthorityStoreError(
                "curriculum authority family ID is invalid"
            )
        with self._guard():
            state = self._read()
            projection = self._project(state).get(family_id)
        return deepcopy(projection) if projection is not None else None

    def active_blueprint(
        self,
        *,
        family_id: str,
        published_revision_id: str,
        published_syllabus_id: str,
        published_syllabus_sha256: str,
        expected_authority_version: int | None = None,
    ) -> dict[str, Any] | None:
        bound = _binding(
            family_id=family_id,
            published_revision_id=published_revision_id,
            published_syllabus_id=published_syllabus_id,
            published_syllabus_sha256=published_syllabus_sha256,
        )
        with self.active_blueprint_lease(
            **bound, expected_authority_version=expected_authority_version
        ) as leased:
            return deepcopy(leased) if leased is not None else None

    @contextmanager
    def active_blueprint_lease(
        self,
        *,
        family_id: str,
        published_revision_id: str,
        published_syllabus_id: str,
        published_syllabus_sha256: str,
        expected_authority_version: int | None = None,
    ) -> Iterator[dict[str, Any] | None]:
        """Hold the cross-process ledger lock through one grading commit.

        Revocation, review, purge, and grading commit therefore have one
        linearizable order instead of a check/use race.
        """

        bound = _binding(
            family_id=family_id,
            published_revision_id=published_revision_id,
            published_syllabus_id=published_syllabus_id,
            published_syllabus_sha256=published_syllabus_sha256,
        )
        if expected_authority_version is not None and (
            isinstance(expected_authority_version, bool)
            or not isinstance(expected_authority_version, int)
            or expected_authority_version < 1
        ):
            raise CurriculumAuthorityStoreError(
                "curriculum authority expected version is invalid"
            )
        with self._guard():
            projection = self._project(self._read()).get(family_id)
            if projection is None:
                yield None
                return
            if (
                expected_authority_version is not None
                and projection["version"] != expected_authority_version
            ):
                raise CurriculumAuthorityStoreError(
                    "curriculum authority version changed"
                )
            if any(
                projection[key] != value
                for key, value in bound.items()
                if key != "family_id"
            ):
                raise CurriculumAuthorityStoreError(
                    "curriculum authority is stale for the published syllabus"
                )
            if projection["status"] != "sealed":
                raise CurriculumAuthorityStoreError(
                    "curriculum authority is not actively sealed"
                )
            blueprint = deepcopy(projection["seal"]["curriculum_blueprint"])
            try:
                verify_teacher_curriculum_runtime_authority(
                    blueprint,
                    trusted_teacher_public_keys=self._trusted_keys(),
                )
            except CurriculumBlueprintError as exc:
                raise CurriculumAuthorityStoreError(
                    "curriculum runtime authority verification failed"
                ) from exc
            yield blueprint

    def export_families(
        self, family_ids: set[str]
    ) -> dict[str, dict[str, Any]]:
        if any(_FAMILY.fullmatch(item) is None for item in family_ids):
            raise CurriculumAuthorityStoreError(
                "curriculum authority export family is invalid"
            )
        with self._guard():
            projections = self._project(self._read())
        return {
            family_id: deepcopy(projections[family_id])
            for family_id in sorted(family_ids)
            if family_id in projections
        }

    def purge_families(
        self, family_ids: list[str], *, occurred_at_utc: str | None = None
    ) -> dict[str, int]:
        if (
            len(family_ids) > _MAX_EVENTS
            or any(_FAMILY.fullmatch(item) is None for item in family_ids)
            or len(family_ids) != len(set(family_ids))
        ):
            raise CurriculumAuthorityStoreError(
                "purged curriculum authority families are invalid"
            )
        selected = set(family_ids)
        now = _utc(occurred_at_utc)
        with self._guard():
            state = self._read()
            projections = self._project(state)
            missing = {
                family_id
                for family_id in selected
                if family_id not in projections
                and _tombstone_key(self.scope_id, family_id)
                not in state["erasure_tombstones"]
            }
            if missing:
                raise CurriculumAuthorityStoreError(
                    "purged curriculum authority family was not found"
                )
            removed = [
                event
                for event in state["events"]
                if event["payload"]["family_id"] in selected
            ]
            for family_id in selected:
                digest = _tombstone_key(self.scope_id, family_id)
                state["erasure_tombstones"].setdefault(
                    digest, {"purged_at_utc": now, "generation": 1}
                )
            retained = [
                deepcopy(event)
                for event in state["events"]
                if event["payload"]["family_id"] not in selected
            ]
            rebuilt: list[dict[str, Any]] = []
            previous: str | None = None
            for sequence, event in enumerate(retained, start=1):
                event["sequence"] = sequence
                event["previous_event_sha256"] = previous
                event["event_id"] = "curae_" + _hash(
                    {
                        "sequence": sequence,
                        "event_type": event["event_type"],
                        "request_sha256": event["request_sha256"],
                        "previous": previous,
                    }
                )[:24]
                event["event_sha256"] = _event_hash(event)
                rebuilt.append(event)
                previous = str(event["event_sha256"])
            state["events"] = rebuilt
            state["event_sequence"] = len(rebuilt)
            self._write(state)
            return {
                "families": len(selected),
                "events": len(removed),
                "tombstones": len(selected),
            }


def curriculum_runtime_authority_projection(
    blueprint: Mapping[str, Any],
    *,
    legacy_lesson_id: str,
    family_id: str,
    published_revision_id: str,
    published_syllabus_id: str,
    published_syllabus_sha256: str,
    authority_version: int,
    trusted_teacher_public_keys: Mapping[str, bytes | str],
) -> dict[str, Any]:
    """Project an exact, content-free runtime authority for one lesson."""

    if (
        isinstance(authority_version, bool)
        or not isinstance(authority_version, int)
        or authority_version < 1
    ):
        raise CurriculumAuthorityStoreError(
            "curriculum runtime authority version is invalid"
        )

    verified = verify_teacher_curriculum_runtime_authority(
        blueprint,
        trusted_teacher_public_keys=trusted_teacher_public_keys,
    )
    lessons = [
        row
        for row in blueprint.get("lessons", [])
        if isinstance(row, Mapping)
        and row.get("legacy_lesson_id") == legacy_lesson_id
    ]
    if len(lessons) != 1:
        raise CurriculumAuthorityStoreError(
            "sealed curriculum does not map the selected syllabus lesson exactly"
        )
    lesson = lessons[0]
    objective_ids = list(lesson["objective_ids"])
    kc_ids = list(lesson["kc_ids"])
    rubrics = [
        row
        for row in blueprint["rubrics"]
        if row["objective_id"] in objective_ids and row["kc_id"] in kc_ids
    ]
    items = [
        row
        for row in blueprint["item_blueprints"]
        if row["objective_id"] in objective_ids and row["kc_id"] in kc_ids
    ]
    if not rubrics or len(items) < 2:
        raise CurriculumAuthorityStoreError(
            "sealed curriculum lesson lacks complete measurement authority"
        )
    factual_claim_ids = [
        row["claim_id"]
        for row in blueprint["factual_claims"]
        if set(row["kc_ids"]).intersection(kc_ids)
    ]
    projection: dict[str, Any] = {
        "schema": CURRICULUM_RUNTIME_AUTHORITY_SCHEMA,
        "authority": True,
        "authoritative_for_runtime_grading": True,
        "curriculum_id": verified["curriculum_id"],
        "curriculum_content_sha256": blueprint["integrity"]["content_sha256"],
        "receipt_id": verified["receipt_id"],
        "receipt_sha256": verified["receipt_sha256"],
        "signing_key_id": verified["signing_key_id"],
        "family_id": family_id,
        "published_revision_id": published_revision_id,
        "published_syllabus_id": published_syllabus_id,
        "published_syllabus_sha256": published_syllabus_sha256,
        "authority_version": authority_version,
        "lesson_id": lesson["lesson_id"],
        "legacy_lesson_id": legacy_lesson_id,
        "objective_ids": objective_ids,
        "kc_ids": kc_ids,
        "factual_claim_ids": factual_claim_ids,
        "rubric_ids": [row["rubric_id"] for row in rubrics],
        "item_blueprint_ids": [row["item_blueprint_id"] for row in items],
    }
    projection["projection_sha256"] = _hash(projection)
    return projection


__all__ = [
    "CURRICULUM_AUTHORITY_EVENT_SCHEMA",
    "CURRICULUM_AUTHORITY_PROJECTION_SCHEMA",
    "CURRICULUM_AUTHORITY_STORE_SCHEMA",
    "CURRICULUM_RUNTIME_AUTHORITY_SCHEMA",
    "CurriculumAuthorityConflictError",
    "CurriculumAuthorityStoreError",
    "TeachingCurriculumAuthorityStore",
    "curriculum_runtime_authority_projection",
]
