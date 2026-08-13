"""Durable, append-only version control for immutable teaching syllabi.

Syllabus JSON documents remain content-addressed and are never overwritten.
This store records which immutable revision is a draft, which one is published,
and every publish/rollback decision.  It intentionally identifies the current
actor as an unauthenticated local operator until the production identity layer
supplies a stronger principal.
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
from typing import Any, Iterator, Mapping

try:  # pragma: no cover - exercised on supported POSIX deployment targets
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from .io_utils import ensure_private_directory


SYLLABUS_VERSION_STORE_SCHEMA = "teaching_skill_miner.syllabus_version_store.v1"
SYLLABUS_VERSION_EVENT_SCHEMA = "teaching_skill_miner.syllabus_version_event.v1"
SYLLABUS_VERSION_PROJECTION_SCHEMA = (
    "teaching_skill_miner.syllabus_version_projection.v1"
)

_SYLLABUS_ID = re.compile(r"^syl_[0-9a-f]{24}$")
_FAMILY_ID = re.compile(r"^syf_[0-9a-f]{24}$")
_REVISION_ID = re.compile(r"^syr_[0-9a-f]{24}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_MAX_EVENTS = 8_192
_MAX_BYTES = 16 * 1024 * 1024


class TeachingSyllabusVersionError(RuntimeError):
    """Raised when a version transition cannot be proven safe."""


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
        raise TeachingSyllabusVersionError(
            "version state must be canonical JSON"
        ) from exc


def _hash(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _utc(value: str | None = None) -> str:
    result = value or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not isinstance(result, str) or not _UTC.fullmatch(result):
        raise TeachingSyllabusVersionError(
            "timestamp must be UTC with second precision"
        )
    try:
        datetime.strptime(result, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise TeachingSyllabusVersionError(
            "timestamp is not a real UTC instant"
        ) from exc
    return result


def _safe_text(value: Any, field: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not 1 <= len(value) <= maximum
    ):
        raise TeachingSyllabusVersionError(
            f"{field} must be a trimmed non-empty string <= {maximum} chars"
        )
    return value


def _actor() -> dict[str, Any]:
    return {
        "actor_kind": "local_operator_not_authenticated",
        "authenticated": False,
        "teacher_identity_claimed": False,
    }


def _empty_state() -> dict[str, Any]:
    return {
        "schema": SYLLABUS_VERSION_STORE_SCHEMA,
        "version": 1,
        "event_sequence": 0,
        "events": [],
        "erasure_tombstones": {},
    }


def _family_tombstone_key(family_id: str) -> str:
    return sha256(family_id.encode("utf-8")).hexdigest()


def _event_hash(event: Mapping[str, Any]) -> str:
    material = deepcopy(dict(event))
    material.pop("event_sha256", None)
    return _hash(material)


def _validate_event(event: Any, *, sequence: int, previous_hash: str | None) -> None:
    if not isinstance(event, Mapping) or set(event) != {
        "schema",
        "sequence",
        "event_id",
        "event_type",
        "occurred_at_utc",
        "idempotency_key",
        "request_sha256",
        "payload",
        "actor",
        "previous_event_sha256",
        "event_sha256",
    }:
        raise TeachingSyllabusVersionError("version event shape is invalid")
    if (
        event["schema"] != SYLLABUS_VERSION_EVENT_SCHEMA
        or event["sequence"] != sequence
    ):
        raise TeachingSyllabusVersionError("version event sequence is invalid")
    if not isinstance(event["event_id"], str) or not event["event_id"].startswith(
        "syve_"
    ):
        raise TeachingSyllabusVersionError("version event ID is invalid")
    if event["event_type"] not in {
        "family.registered",
        "revision.created",
        "revision.published",
        "revision.rollback",
    }:
        raise TeachingSyllabusVersionError("version event type is invalid")
    _utc(event["occurred_at_utc"])
    if not _IDEMPOTENCY.fullmatch(str(event["idempotency_key"])):
        raise TeachingSyllabusVersionError("version idempotency key is invalid")
    if not _HEX64.fullmatch(str(event["request_sha256"])):
        raise TeachingSyllabusVersionError("version request hash is invalid")
    if not isinstance(event["payload"], Mapping):
        raise TeachingSyllabusVersionError("version event payload is invalid")
    if dict(event["actor"]) != _actor():
        raise TeachingSyllabusVersionError("version actor boundary is invalid")
    if event["previous_event_sha256"] != previous_hash:
        raise TeachingSyllabusVersionError("version event hash chain is invalid")
    if event["event_sha256"] != _event_hash(event):
        raise TeachingSyllabusVersionError("version event hash is invalid")


def _validate_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping) or set(state) != {
        "schema",
        "version",
        "event_sequence",
        "events",
        "erasure_tombstones",
    }:
        raise TeachingSyllabusVersionError("version store state is invalid")
    if state["schema"] != SYLLABUS_VERSION_STORE_SCHEMA or state["version"] != 1:
        raise TeachingSyllabusVersionError("version store schema is invalid")
    events = state["events"]
    if not isinstance(events, list) or len(events) > _MAX_EVENTS:
        raise TeachingSyllabusVersionError("version event capacity is invalid")
    if state["event_sequence"] != len(events):
        raise TeachingSyllabusVersionError("version event head is invalid")
    tombstones = state["erasure_tombstones"]
    if not isinstance(tombstones, Mapping) or len(tombstones) > _MAX_EVENTS:
        raise TeachingSyllabusVersionError("version erasure tombstones are invalid")
    for digest, tombstone in tombstones.items():
        if not _HEX64.fullmatch(str(digest)) or not isinstance(tombstone, Mapping):
            raise TeachingSyllabusVersionError("version erasure tombstone is invalid")
        if set(tombstone) != {"purged_at_utc", "generation"}:
            raise TeachingSyllabusVersionError(
                "version erasure tombstone shape is invalid"
            )
        _utc(str(tombstone["purged_at_utc"]))
        generation = tombstone["generation"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            raise TeachingSyllabusVersionError("version erasure generation is invalid")
    previous: str | None = None
    idempotency: dict[str, str] = {}
    for sequence, event in enumerate(events, start=1):
        _validate_event(event, sequence=sequence, previous_hash=previous)
        key = str(event["idempotency_key"])
        request_hash = str(event["request_sha256"])
        if key in idempotency and idempotency[key] != request_hash:
            raise TeachingSyllabusVersionError(
                "idempotency key has conflicting content"
            )
        idempotency[key] = request_hash
        previous = str(event["event_sha256"])
    canonical = _canonical(state)
    if len(canonical) > _MAX_BYTES:
        raise TeachingSyllabusVersionError("version store exceeds its byte limit")
    projection = _project(state)
    if any(
        _family_tombstone_key(family_id) in tombstones
        for family_id in projection["families"]
    ):
        raise TeachingSyllabusVersionError(
            "erased syllabus family was resurrected in the version ledger"
        )
    return deepcopy(dict(state))


def _project(state: Mapping[str, Any]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    syllabus_to_family: dict[str, str] = {}
    for event in state["events"]:
        payload = event["payload"]
        family_id = str(payload.get("family_id", ""))
        if not _FAMILY_ID.fullmatch(family_id):
            raise TeachingSyllabusVersionError("family_id is invalid")
        event_type = event["event_type"]
        if event_type == "family.registered":
            if family_id in families:
                raise TeachingSyllabusVersionError("family was registered twice")
            revision = _revision_from_payload(payload)
            if revision["revision_number"] != 1:
                raise TeachingSyllabusVersionError(
                    "initial revision number must be one"
                )
            families[family_id] = {
                "family_id": family_id,
                "version": 1,
                "published_revision_id": revision["revision_id"],
                "revisions": [revision],
                "history": [str(event["event_id"])],
            }
            syllabus_to_family[revision["syllabus_id"]] = family_id
            continue
        family = families.get(family_id)
        if family is None:
            raise TeachingSyllabusVersionError(
                "version event references unknown family"
            )
        if event_type == "revision.created":
            revision = _revision_from_payload(payload)
            if revision["revision_number"] != len(family["revisions"]) + 1:
                raise TeachingSyllabusVersionError("revision number is not contiguous")
            if revision["parent_revision_id"] not in {
                row["revision_id"] for row in family["revisions"]
            }:
                raise TeachingSyllabusVersionError("revision parent is unknown")
            if revision["syllabus_id"] in syllabus_to_family:
                raise TeachingSyllabusVersionError(
                    "syllabus belongs to another revision"
                )
            family["revisions"].append(revision)
            syllabus_to_family[revision["syllabus_id"]] = family_id
        else:
            revision_id = str(payload.get("revision_id", ""))
            if revision_id not in {row["revision_id"] for row in family["revisions"]}:
                raise TeachingSyllabusVersionError("published revision is unknown")
            if (
                event_type == "revision.rollback"
                and revision_id == family["published_revision_id"]
            ):
                raise TeachingSyllabusVersionError(
                    "rollback target is already published"
                )
            family["published_revision_id"] = revision_id
        family["version"] += 1
        family["history"].append(str(event["event_id"]))
    return {"families": families, "syllabus_to_family": syllabus_to_family}


def _revision_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    revision_id = str(payload.get("revision_id", ""))
    syllabus_id = str(payload.get("syllabus_id", ""))
    content_sha256 = str(payload.get("content_sha256", ""))
    revision_number = payload.get("revision_number")
    parent = payload.get("parent_revision_id")
    if not _REVISION_ID.fullmatch(revision_id):
        raise TeachingSyllabusVersionError("revision_id is invalid")
    if not _SYLLABUS_ID.fullmatch(syllabus_id) or not _HEX64.fullmatch(content_sha256):
        raise TeachingSyllabusVersionError("revision syllabus identity is invalid")
    if (
        isinstance(revision_number, bool)
        or not isinstance(revision_number, int)
        or revision_number < 1
    ):
        raise TeachingSyllabusVersionError("revision_number is invalid")
    if parent is not None and not _REVISION_ID.fullmatch(str(parent)):
        raise TeachingSyllabusVersionError("parent_revision_id is invalid")
    summary = _safe_text(payload.get("change_summary"), "change_summary", maximum=500)
    return {
        "revision_id": revision_id,
        "revision_number": revision_number,
        "syllabus_id": syllabus_id,
        "content_sha256": content_sha256,
        "parent_revision_id": parent,
        "created_at_utc": _utc(str(payload.get("created_at_utc", ""))),
        "change_summary": summary,
    }


def _public_family(family: Mapping[str, Any]) -> dict[str, Any]:
    published = str(family["published_revision_id"])
    return {
        "schema": SYLLABUS_VERSION_PROJECTION_SCHEMA,
        "family_id": str(family["family_id"]),
        "version": int(family["version"]),
        "published_revision_id": published,
        "revisions": [
            {
                **deepcopy(row),
                "status": (
                    "published"
                    if row["revision_id"] == published
                    else "draft"
                    if int(row["revision_number"])
                    == max(int(item["revision_number"]) for item in family["revisions"])
                    else "historical"
                ),
            }
            for row in family["revisions"]
        ],
        "actor_boundary": _actor(),
    }


class TeachingSyllabusVersionStore:
    """Cross-process-safe immutable syllabus version ledger."""

    def __init__(self, root: str | Path) -> None:
        candidate = Path(root).expanduser()
        if candidate.exists() and candidate.is_symlink():
            raise TeachingSyllabusVersionError("version store root cannot be a symlink")
        self.root = ensure_private_directory(candidate).resolve()
        self.path = self.root / "syllabus_versions.json"
        self.lock_path = self.root / ".syllabus_versions.lock"
        self._lock = threading.RLock()

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

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_state()
        if self.path.is_symlink() or not self.path.is_file():
            raise TeachingSyllabusVersionError("version store path is unsafe")
        try:
            raw = self.path.read_bytes()
            if len(raw) > _MAX_BYTES:
                raise TeachingSyllabusVersionError(
                    "version store exceeds its byte limit"
                )
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TeachingSyllabusVersionError("version store cannot be read") from exc
        # v1 ledgers created before erasure fencing had no tombstone member.
        # Upgrade in memory; the next mutation persists the strict new shape.
        if isinstance(value, dict) and set(value) == {
            "schema",
            "version",
            "event_sequence",
            "events",
        }:
            value["erasure_tombstones"] = {}
        return _validate_state(value)

    def _write(self, state: Mapping[str, Any]) -> None:
        candidate = _validate_state(state)
        payload = _canonical(candidate) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".syllabus_versions.", suffix=".tmp", dir=self.root
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
        occurred_at_utc: str,
    ) -> tuple[dict[str, Any], bool]:
        if not _IDEMPOTENCY.fullmatch(idempotency_key):
            raise TeachingSyllabusVersionError("idempotency_key is invalid")
        request_hash = _hash({"event_type": event_type, "payload": payload})
        for event in state["events"]:
            if event["idempotency_key"] != idempotency_key:
                continue
            if event["request_sha256"] != request_hash:
                raise TeachingSyllabusVersionError(
                    "idempotency_key was reused with different content"
                )
            return state, False
        if len(state["events"]) >= _MAX_EVENTS:
            raise TeachingSyllabusVersionError("version event capacity is exhausted")
        sequence = len(state["events"]) + 1
        previous = state["events"][-1]["event_sha256"] if state["events"] else None
        event = {
            "schema": SYLLABUS_VERSION_EVENT_SCHEMA,
            "sequence": sequence,
            "event_id": "syve_"
            + _hash(
                {
                    "sequence": sequence,
                    "event_type": event_type,
                    "request_sha256": request_hash,
                    "previous": previous,
                }
            )[:24],
            "event_type": event_type,
            "occurred_at_utc": occurred_at_utc,
            "idempotency_key": idempotency_key,
            "request_sha256": request_hash,
            "payload": deepcopy(dict(payload)),
            "actor": _actor(),
            "previous_event_sha256": previous,
        }
        event["event_sha256"] = _event_hash(event)
        state["events"].append(event)
        state["event_sequence"] = sequence
        self._write(state)
        return state, True

    def register(
        self,
        syllabus: Mapping[str, Any],
        *,
        idempotency_key: str,
        occurred_at_utc: str | None = None,
    ) -> dict[str, Any]:
        syllabus_id, content_hash = self._syllabus_identity(syllabus)
        now = _utc(occurred_at_utc)
        family_id = "syf_" + _hash({"root_syllabus_id": syllabus_id})[:24]
        revision_id = (
            "syr_"
            + _hash(
                {"family_id": family_id, "syllabus_id": syllabus_id, "revision": 1}
            )[:24]
        )
        payload = {
            "family_id": family_id,
            "revision_id": revision_id,
            "revision_number": 1,
            "syllabus_id": syllabus_id,
            "content_sha256": content_hash,
            "parent_revision_id": None,
            "created_at_utc": now,
            "change_summary": "Initial immutable syllabus revision",
        }
        with self._guard():
            state = self._read()
            projection = _project(state)
            if _family_tombstone_key(family_id) in state["erasure_tombstones"]:
                raise TeachingSyllabusVersionError(
                    "syllabus family was permanently erased"
                )
            replay = next(
                (
                    event
                    for event in state["events"]
                    if event["idempotency_key"] == idempotency_key
                ),
                None,
            )
            if replay is not None and (
                replay["event_type"] != "family.registered"
                or replay["payload"].get("syllabus_id") != syllabus_id
            ):
                raise TeachingSyllabusVersionError(
                    "idempotency_key was reused with different content"
                )
            existing_family = projection["syllabus_to_family"].get(syllabus_id)
            if existing_family is not None:
                if existing_family != family_id:
                    raise TeachingSyllabusVersionError(
                        "syllabus family identity conflicts"
                    )
                return _public_family(projection["families"][family_id])
            state, _ = self._append(
                state,
                event_type="family.registered",
                payload=payload,
                idempotency_key=idempotency_key,
                occurred_at_utc=now,
            )
            return _public_family(_project(state)["families"][family_id])

    def create_revision(
        self,
        *,
        base_syllabus_id: str,
        revised_syllabus: Mapping[str, Any],
        change_summary: str,
        expected_version: int,
        idempotency_key: str,
        occurred_at_utc: str | None = None,
    ) -> dict[str, Any]:
        revised_id, content_hash = self._syllabus_identity(revised_syllabus)
        summary = _safe_text(change_summary, "change_summary", maximum=500)
        now = _utc(occurred_at_utc)
        with self._guard():
            state = self._read()
            projection = _project(state)
            family_id = projection["syllabus_to_family"].get(base_syllabus_id)
            if family_id is None:
                raise TeachingSyllabusVersionError("base syllabus is not versioned")
            family = projection["families"][family_id]
            replay = next(
                (
                    event
                    for event in state["events"]
                    if event["idempotency_key"] == idempotency_key
                ),
                None,
            )
            if replay is not None:
                replay_payload = replay["payload"]
                if (
                    replay["event_type"] != "revision.created"
                    or replay_payload.get("family_id") != family_id
                    or replay_payload.get("syllabus_id") != revised_id
                    or replay_payload.get("content_sha256") != content_hash
                    or replay_payload.get("change_summary") != summary
                ):
                    raise TeachingSyllabusVersionError(
                        "idempotency_key was reused with different content"
                    )
                return _public_family(family)
            if family["version"] != expected_version:
                raise TeachingSyllabusVersionError("syllabus family version conflict")
            if revised_id in projection["syllabus_to_family"]:
                raise TeachingSyllabusVersionError(
                    "revised syllabus is already versioned"
                )
            revision_number = len(family["revisions"]) + 1
            parent_revision = next(
                row
                for row in family["revisions"]
                if row["syllabus_id"] == base_syllabus_id
            )
            revision_id = (
                "syr_"
                + _hash(
                    {
                        "family_id": family_id,
                        "syllabus_id": revised_id,
                        "revision": revision_number,
                    }
                )[:24]
            )
            payload = {
                "family_id": family_id,
                "revision_id": revision_id,
                "revision_number": revision_number,
                "syllabus_id": revised_id,
                "content_sha256": content_hash,
                "parent_revision_id": parent_revision["revision_id"],
                "created_at_utc": now,
                "change_summary": summary,
            }
            state, _ = self._append(
                state,
                event_type="revision.created",
                payload=payload,
                idempotency_key=idempotency_key,
                occurred_at_utc=now,
            )
            return _public_family(_project(state)["families"][family_id])

    def publish(
        self,
        *,
        family_id: str,
        revision_id: str,
        expected_version: int,
        idempotency_key: str,
        occurred_at_utc: str | None = None,
    ) -> dict[str, Any]:
        return self._point(
            event_type="revision.published",
            family_id=family_id,
            revision_id=revision_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            occurred_at_utc=occurred_at_utc,
        )

    def rollback(
        self,
        *,
        family_id: str,
        revision_id: str,
        expected_version: int,
        idempotency_key: str,
        occurred_at_utc: str | None = None,
    ) -> dict[str, Any]:
        return self._point(
            event_type="revision.rollback",
            family_id=family_id,
            revision_id=revision_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            occurred_at_utc=occurred_at_utc,
        )

    def _point(
        self,
        *,
        event_type: str,
        family_id: str,
        revision_id: str,
        expected_version: int,
        idempotency_key: str,
        occurred_at_utc: str | None,
    ) -> dict[str, Any]:
        if not _FAMILY_ID.fullmatch(family_id) or not _REVISION_ID.fullmatch(
            revision_id
        ):
            raise TeachingSyllabusVersionError("family or revision ID is invalid")
        now = _utc(occurred_at_utc)
        payload = {"family_id": family_id, "revision_id": revision_id}
        with self._guard():
            state = self._read()
            family = _project(state)["families"].get(family_id)
            if family is None:
                raise TeachingSyllabusVersionError("syllabus family was not found")
            existing = next(
                (
                    event
                    for event in state["events"]
                    if event["idempotency_key"] == idempotency_key
                ),
                None,
            )
            if existing is None and family["version"] != expected_version:
                raise TeachingSyllabusVersionError("syllabus family version conflict")
            state, _ = self._append(
                state,
                event_type=event_type,
                payload=payload,
                idempotency_key=idempotency_key,
                occurred_at_utc=now,
            )
            return _public_family(_project(state)["families"][family_id])

    def read_family(self, identity: str) -> dict[str, Any]:
        with self._guard():
            projection = _project(self._read())
        family_id = (
            identity
            if _FAMILY_ID.fullmatch(identity)
            else projection["syllabus_to_family"].get(identity)
        )
        if family_id is None or family_id not in projection["families"]:
            raise TeachingSyllabusVersionError("syllabus family was not found")
        return _public_family(projection["families"][family_id])

    def list_families(self) -> list[dict[str, Any]]:
        with self._guard():
            families = _project(self._read())["families"]
        return [_public_family(families[key]) for key in sorted(families)]

    def idempotency_event(self, idempotency_key: str) -> dict[str, Any] | None:
        """Return a content-safe immutable event receipt for exact retry recovery."""

        if not _IDEMPOTENCY.fullmatch(idempotency_key):
            raise TeachingSyllabusVersionError("idempotency_key is invalid")
        with self._guard():
            state = self._read()
        matches = [
            event
            for event in state["events"]
            if event["idempotency_key"] == idempotency_key
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise TeachingSyllabusVersionError(
                "idempotency key has multiple durable events"
            )
        event = matches[0]
        return {
            "event_type": str(event["event_type"]),
            "occurred_at_utc": str(event["occurred_at_utc"]),
            "request_sha256": str(event["request_sha256"]),
            "payload": deepcopy(dict(event["payload"])),
            "event_sha256": str(event["event_sha256"]),
        }

    def purge_families(
        self,
        family_ids: list[str],
        *,
        occurred_at_utc: str | None = None,
    ) -> dict[str, int]:
        """Physically compact family events after first fsyncing exact fences.

        Tombstones retain only a hash of each family identity plus time/generation;
        no title, lesson, resource, syllabus ID, or operator-authored text survives.
        """

        if (
            not isinstance(family_ids, list)
            or len(family_ids) > _MAX_EVENTS
            or any(
                not isinstance(item, str) or not _FAMILY_ID.fullmatch(item)
                for item in family_ids
            )
            or len(family_ids) != len(set(family_ids))
        ):
            raise TeachingSyllabusVersionError("purged family IDs are invalid")
        now = _utc(occurred_at_utc)
        selected = set(family_ids)
        with self._guard():
            state = self._read()
            projection = _project(state)
            missing = selected.difference(projection["families"])
            missing_without_fence = {
                family_id
                for family_id in missing
                if _family_tombstone_key(family_id) not in state["erasure_tombstones"]
            }
            if missing_without_fence:
                raise TeachingSyllabusVersionError(
                    "purged syllabus family was not found"
                )
            removed = [
                deepcopy(event)
                for event in state["events"]
                if str(event["payload"].get("family_id")) in selected
            ]
            for family_id in selected:
                digest = _family_tombstone_key(family_id)
                existing = state["erasure_tombstones"].get(digest)
                if existing is None:
                    state["erasure_tombstones"][digest] = {
                        "purged_at_utc": now,
                        "generation": 1,
                    }
            retained = [
                deepcopy(event)
                for event in state["events"]
                if str(event["payload"].get("family_id")) not in selected
            ]
            rebuilt: list[dict[str, Any]] = []
            previous: str | None = None
            for sequence, source in enumerate(retained, start=1):
                event = deepcopy(source)
                event["sequence"] = sequence
                event["previous_event_sha256"] = previous
                event["event_id"] = (
                    "syve_"
                    + _hash(
                        {
                            "sequence": sequence,
                            "event_type": event["event_type"],
                            "request_sha256": event["request_sha256"],
                            "previous": previous,
                        }
                    )[:24]
                )
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

    def require_published(self, syllabus_id: str) -> dict[str, Any]:
        family = self.read_family(syllabus_id)
        published = next(
            row
            for row in family["revisions"]
            if row["revision_id"] == family["published_revision_id"]
        )
        if published["syllabus_id"] != syllabus_id:
            raise TeachingSyllabusVersionError(
                "only the currently published syllabus revision may start a new lesson"
            )
        return family

    @contextmanager
    def published_revision_lease(
        self,
        *,
        family_id: str,
        revision_id: str,
        syllabus_id: str,
        content_sha256: str,
    ) -> Iterator[dict[str, Any]]:
        """Fence publish/rollback while a bound grading commit is persisted."""

        with self._guard():
            projection = _project(self._read())
            family = projection["families"].get(family_id)
            if family is None or family["published_revision_id"] != revision_id:
                raise TeachingSyllabusVersionError(
                    "published syllabus revision changed"
                )
            matches = [
                row
                for row in family["revisions"]
                if row["revision_id"] == revision_id
                and row["syllabus_id"] == syllabus_id
                and row["content_sha256"] == content_sha256
            ]
            if len(matches) != 1:
                raise TeachingSyllabusVersionError(
                    "published syllabus content binding changed"
                )
            yield _public_family(family)

    @staticmethod
    def _syllabus_identity(syllabus: Mapping[str, Any]) -> tuple[str, str]:
        from .teacher_agent_syllabus import (  # noqa: PLC0415
            validate_teaching_syllabus,
        )

        validate_teaching_syllabus(syllabus)
        return str(syllabus["syllabus_id"]), str(
            syllabus["integrity"]["content_sha256"]
        )
