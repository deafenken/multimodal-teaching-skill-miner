"""Authenticated, context-only review projections for teaching resources.

The extractor deliberately blocks OCR, visual, temporal, and cross-layer
conflicts from session use.  This module provides the only downgrade path from
that block: an authenticated teacher must submit a bounded transcription that
they actually compared with the original source.  A review never creates an
answer key and never authorizes grading or mastery updates.

The durable store is intentionally separate from the immutable raw-content
index.  The original content hash remains the resource identity while every
reviewed projection has its own append-only revision and hash-bound receipt.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Any, Iterator, Mapping

from .teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    TeacherAuthorityError,
    canonical_sha256,
    validate_teacher_authority_verification_receipt,
)
from .teacher_agent_resources import (
    MAX_RESOURCE_TEXT_CHARS,
    TeachingResourceError,
    validate_teaching_resources,
)


RESOURCE_REVIEW_PROJECTION_SCHEMA = "teaching_skill_miner.resource_review_projection.v1"
RESOURCE_REVIEW_RECEIPT_SCHEMA = "teaching_skill_miner.resource_review_receipt.v1"
RESOURCE_REVIEW_STORE_SCHEMA = "teaching_skill_miner.resource_review_store.v1"
RESOURCE_REVIEW_REVISION_SCHEMA = "teaching_skill_miner.resource_review_revision.v1"

MAX_REVIEW_REVISIONS = 16
MAX_REVIEW_NOTE_CHARS = 1_000
MAX_REVIEW_STORE_BYTES = 1_000_000

_RESOURCE_ID = re.compile(r"res_[0-9a-f]{20}")
_STAGE_ID = re.compile(r"stage_[0-9a-f]{24}")
_CONTENT_HASH = re.compile(r"[0-9a-f]{64}")
_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}")
_PROJECTION_ID = re.compile(r"rrp_[0-9a-f]{24}")
_PENDING_MARKERS = (
    "[内容冲突待确认：",
    "[视觉复核：",
    "[公式转写候选：",
    "[扫描页检测：",
)


class TeachingResourceReviewError(ValueError):
    """Raised when a resource review cannot be trusted or persisted."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise TeachingResourceReviewError(
            "resource review must contain canonical JSON"
        ) from exc


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_timestamp(value: datetime | None) -> str:
    current = value or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise TeachingResourceReviewError("resource review time must be timezone-aware")
    return (
        current.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _descriptor_material(resource: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(dict(resource))
    value.pop("staged_resource_id", None)
    value.pop("review_projection", None)
    value.pop("retrieval_index", None)
    value.pop("retrieval_indexed_char_count", None)
    value.pop("retrieval_index_truncated", None)
    return value


def resource_descriptor_sha256(resource: Mapping[str, Any]) -> str:
    """Hash the immutable extraction projection without staging-only fields."""

    try:
        validate_teaching_resources([_descriptor_material(resource)])
    except TeachingResourceError as exc:
        raise TeachingResourceReviewError(
            "original teaching resource is invalid"
        ) from exc
    return canonical_sha256(_descriptor_material(resource))


def _validated_review_request(
    request: Mapping[str, Any], *, original: Mapping[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "resource_id",
        "staged_resource_id",
        "content_sha256",
        "expected_review_version",
        "resource_review_idempotency_key",
        "original_resource_sha256",
        "reviewed_text",
        "resolved_conflict_ids",
        "excluded_layer_ids",
        "attestations",
        "review_note",
    }
    if not isinstance(request, Mapping) or set(request) != expected_fields:
        raise TeachingResourceReviewError("resource review request shape is invalid")
    value = _json_copy(dict(request))
    if (
        _RESOURCE_ID.fullmatch(str(value.get("resource_id", ""))) is None
        or _STAGE_ID.fullmatch(str(value.get("staged_resource_id", ""))) is None
        or _CONTENT_HASH.fullmatch(str(value.get("content_sha256", ""))) is None
        or _IDEMPOTENCY_KEY.fullmatch(
            str(value.get("resource_review_idempotency_key", ""))
        )
        is None
        or _CONTENT_HASH.fullmatch(str(value.get("original_resource_sha256", "")))
        is None
        or isinstance(value.get("expected_review_version"), bool)
        or not isinstance(value.get("expected_review_version"), int)
        or not 0 <= int(value["expected_review_version"]) < MAX_REVIEW_REVISIONS
    ):
        raise TeachingResourceReviewError("resource review identity is invalid")
    if (
        value["resource_id"] != original.get("resource_id")
        or value["content_sha256"] != original.get("content_sha256")
        or value["staged_resource_id"]
        != f"stage_{str(original.get('content_sha256', ''))[:24]}"
        or value["original_resource_sha256"] != resource_descriptor_sha256(original)
    ):
        raise TeachingResourceReviewError("resource review original binding changed")

    reviewed_text = value.get("reviewed_text")
    if (
        not isinstance(reviewed_text, str)
        or reviewed_text != reviewed_text.strip()
        or not reviewed_text
        or "\x00" in reviewed_text
        or len(reviewed_text) > MAX_RESOURCE_TEXT_CHARS
        or any(marker in reviewed_text for marker in _PENDING_MARKERS)
    ):
        raise TeachingResourceReviewError(
            "resource review must provide bounded text without unresolved markers"
        )

    contract = original.get("evidence_contract")
    if not isinstance(contract, Mapping):
        raise TeachingResourceReviewError(
            "resource review evidence contract is missing"
        )
    conflicts = contract.get("conflicts", [])
    layers = contract.get("layers", [])
    if not isinstance(conflicts, list) or not isinstance(layers, list):
        raise TeachingResourceReviewError(
            "resource review evidence contract is invalid"
        )
    conflict_ids = [
        str(item.get("conflict_id", ""))
        for item in conflicts
        if isinstance(item, Mapping)
    ]
    resolved = value.get("resolved_conflict_ids")
    if (
        not isinstance(resolved, list)
        or any(not isinstance(item, str) for item in resolved)
        or len(resolved) != len(set(resolved))
        or sorted(resolved) != sorted(conflict_ids)
    ):
        raise TeachingResourceReviewError(
            "resource review must explicitly resolve every source conflict"
        )
    known_layers = {
        str(item.get("layer_id"))
        for item in layers
        if isinstance(item, Mapping) and isinstance(item.get("layer_id"), str)
    }
    excluded = value.get("excluded_layer_ids")
    if (
        not isinstance(excluded, list)
        or any(not isinstance(item, str) for item in excluded)
        or len(excluded) != len(set(excluded))
        or not set(excluded).issubset(known_layers)
    ):
        raise TeachingResourceReviewError("resource review excluded layers are invalid")
    attestations = value.get("attestations")
    if not isinstance(attestations, Mapping) or dict(attestations) != {
        "compared_with_original_source": True,
        "uncertainties_removed_or_explicit": True,
        "not_an_answer_key": True,
        "context_only": True,
    }:
        raise TeachingResourceReviewError("resource review attestations are incomplete")
    note = value.get("review_note")
    if (
        not isinstance(note, str)
        or note != note.strip()
        or len(note) > MAX_REVIEW_NOTE_CHARS
    ):
        raise TeachingResourceReviewError("resource review note is invalid")
    return value


def _validated_authority_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = validate_teacher_authority_verification_receipt(receipt)
    except TeacherAuthorityError as exc:
        raise TeachingResourceReviewError(
            "authenticated teacher authority receipt is invalid"
        ) from exc
    if (
        value.get("authority_kind") != AUTHENTICATED_TEACHER_ACTOR
        or value.get("method") != "POST"
        or value.get("path") != "api/resource/review"
        or value.get("personal_non_repudiation") is not False
    ):
        raise TeachingResourceReviewError(
            "authenticated teacher authority is not bound to resource review"
        )
    return value


def _reviewed_evidence_contract(
    original: Mapping[str, Any], *, excluded_layer_ids: set[str]
) -> dict[str, Any]:
    contract = _json_copy(original.get("evidence_contract"))
    if not isinstance(contract, dict):
        raise TeachingResourceReviewError("resource evidence contract is invalid")
    layers = contract.get("layers")
    if not isinstance(layers, list):
        raise TeachingResourceReviewError("resource evidence layers are invalid")
    for layer in layers:
        if not isinstance(layer, dict):
            raise TeachingResourceReviewError("resource evidence layer is invalid")
        if layer.get("layer_id") in excluded_layer_ids:
            layer["status"] = "unavailable"
        layer["semantic_understanding_established"] = False
        layer["grading_evidence_allowed"] = False
        layer["mastery_evidence_allowed"] = False
    contract.update(
        {
            "conflicts": [],
            "decision": "usable_as_untrusted_teaching_context",
            "transcription_is_semantic_understanding": False,
            "semantic_analysis_is_answer_correctness": False,
            "grading_evidence_allowed": False,
            "mastery_evidence_allowed": False,
        }
    )
    return contract


def build_reviewed_resource_projection(
    original_resource: Mapping[str, Any],
    review_request: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    *,
    review_version: int,
    previous_review_sha256: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build and fully validate one authenticated context-only projection."""

    original = _descriptor_material(original_resource)
    try:
        validate_teaching_resources([original])
    except TeachingResourceError as exc:
        raise TeachingResourceReviewError(
            "original teaching resource is invalid"
        ) from exc
    request = _validated_review_request(review_request, original=original)
    authority = _validated_authority_receipt(authority_receipt)
    if authority.get("body_sha256") != canonical_sha256(request):
        raise TeachingResourceReviewError(
            "teacher authority receipt is not bound to this review request"
        )
    if review_version != int(request["expected_review_version"]) + 1:
        raise TeachingResourceReviewError("resource review version is invalid")
    if (
        previous_review_sha256 is not None
        and _CONTENT_HASH.fullmatch(str(previous_review_sha256)) is None
    ):
        raise TeachingResourceReviewError("resource review predecessor is invalid")

    reviewed_text = str(request["reviewed_text"])
    reviewed = _json_copy(original)
    reviewed.pop("retrieval_index", None)
    reviewed.update(
        {
            "extracted_text": reviewed_text,
            "extracted_char_count": len(reviewed_text),
            "original_extracted_char_count": len(reviewed_text),
            "truncated": False,
            "extraction_engine": "authenticated_teacher_review_projection_v1",
            "needs_review": False,
            "requires_confirmation": False,
            "grading_evidence_allowed": False,
            "mastery_evidence_allowed": False,
            "evidence_contract": _reviewed_evidence_contract(
                original,
                excluded_layer_ids=set(request["excluded_layer_ids"]),
            ),
        }
    )
    created_at = _utc_timestamp(now)
    receipt_material = {
        "schema": RESOURCE_REVIEW_RECEIPT_SCHEMA,
        "review_version": review_version,
        "created_at": created_at,
        "resource_id": reviewed["resource_id"],
        "content_sha256": reviewed["content_sha256"],
        "original_resource_sha256": request["original_resource_sha256"],
        "original_extracted_text_sha256": _text_sha256(str(original["extracted_text"])),
        "original_evidence_contract_sha256": canonical_sha256(
            original["evidence_contract"]
        ),
        "reviewed_text_sha256": _text_sha256(reviewed_text),
        "reviewed_evidence_contract_sha256": canonical_sha256(
            reviewed["evidence_contract"]
        ),
        "resolved_conflict_ids": list(request["resolved_conflict_ids"]),
        "excluded_layer_ids": list(request["excluded_layer_ids"]),
        "attestations": deepcopy(request["attestations"]),
        "review_note_sha256": _text_sha256(str(request["review_note"])),
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "authority_id": authority["authority_id"],
        "authority_receipt_sha256": authority["receipt_sha256"],
        "actor_principal_sha256": authority["actor_principal_sha256"],
        "roles_sha256": authority["roles_sha256"],
        "assurance": authority["assurance"],
        "personal_non_repudiation": False,
        "review_scope": "untrusted_teaching_context_only",
        "semantic_understanding_established": False,
        "grading_evidence_allowed": False,
        "mastery_evidence_allowed": False,
        "previous_review_sha256": previous_review_sha256,
    }
    receipt_material["review_id"] = "rrp_" + canonical_sha256(receipt_material)[:24]
    receipt = {**receipt_material, "receipt_sha256": canonical_sha256(receipt_material)}
    reviewed["review_projection"] = {
        "schema": RESOURCE_REVIEW_PROJECTION_SCHEMA,
        "receipt": receipt,
    }
    validate_reviewed_resource_projection(reviewed)
    return reviewed


def validate_reviewed_resource_projection(
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a stored projection without elevating its trust boundary."""

    value = _json_copy(dict(resource))
    projection = value.pop("review_projection", None)
    try:
        validate_teaching_resources([value])
    except TeachingResourceError as exc:
        raise TeachingResourceReviewError(
            "reviewed resource descriptor is invalid"
        ) from exc
    if (
        not isinstance(projection, Mapping)
        or projection.get("schema") != RESOURCE_REVIEW_PROJECTION_SCHEMA
        or set(projection) != {"schema", "receipt"}
        or not isinstance(projection.get("receipt"), Mapping)
    ):
        raise TeachingResourceReviewError("resource review projection is invalid")
    receipt = dict(projection["receipt"])
    expected_receipt_fields = {
        "schema",
        "review_id",
        "review_version",
        "created_at",
        "resource_id",
        "content_sha256",
        "original_resource_sha256",
        "original_extracted_text_sha256",
        "original_evidence_contract_sha256",
        "reviewed_text_sha256",
        "reviewed_evidence_contract_sha256",
        "resolved_conflict_ids",
        "excluded_layer_ids",
        "attestations",
        "review_note_sha256",
        "authority_kind",
        "authority_id",
        "authority_receipt_sha256",
        "actor_principal_sha256",
        "roles_sha256",
        "assurance",
        "personal_non_repudiation",
        "review_scope",
        "semantic_understanding_established",
        "grading_evidence_allowed",
        "mastery_evidence_allowed",
        "previous_review_sha256",
        "receipt_sha256",
    }
    declared = receipt.pop("receipt_sha256", None)
    if (
        set(receipt) != expected_receipt_fields - {"receipt_sha256"}
        or not isinstance(declared, str)
        or not hmac_compare(declared, canonical_sha256(receipt))
        or receipt.get("schema") != RESOURCE_REVIEW_RECEIPT_SCHEMA
        or _PROJECTION_ID.fullmatch(str(receipt.get("review_id", ""))) is None
        or receipt.get("resource_id") != value.get("resource_id")
        or receipt.get("content_sha256") != value.get("content_sha256")
        or receipt.get("reviewed_text_sha256")
        != _text_sha256(str(value.get("extracted_text", "")))
        or receipt.get("reviewed_evidence_contract_sha256")
        != canonical_sha256(value.get("evidence_contract"))
        or receipt.get("authority_kind") != AUTHENTICATED_TEACHER_ACTOR
        or receipt.get("personal_non_repudiation") is not False
        or receipt.get("review_scope") != "untrusted_teaching_context_only"
        or receipt.get("semantic_understanding_established") is not False
        or receipt.get("grading_evidence_allowed") is not False
        or receipt.get("mastery_evidence_allowed") is not False
        or value.get("needs_review") is not False
        or value.get("requires_confirmation") is not False
        or value.get("grading_evidence_allowed") is not False
        or value.get("mastery_evidence_allowed") is not False
        or value.get("evidence_contract", {}).get("decision")
        != "usable_as_untrusted_teaching_context"
        or value.get("evidence_contract", {}).get("conflicts") != []
    ):
        raise TeachingResourceReviewError("resource review receipt is invalid")
    return {**value, "review_projection": _json_copy(projection)}


def hmac_compare(left: str, right: str) -> bool:
    """Constant-time digest comparison without accepting non-digests."""

    return bool(
        _CONTENT_HASH.fullmatch(str(left))
        and _CONTENT_HASH.fullmatch(str(right))
        and hmac.compare_digest(str(left), str(right))
    )


class TeachingResourceReviewStore:
    """Private, atomic, cross-process revision store for reviewed projections."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        candidate = Path(root)
        if candidate.exists() and candidate.is_symlink():
            raise TeachingResourceReviewError(
                "resource review root cannot be a symlink"
            )
        candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            candidate.chmod(0o700)
        except OSError as exc:
            raise TeachingResourceReviewError(
                "resource review root permissions are invalid"
            ) from exc
        self.root = candidate.resolve()
        self._thread_lock = RLock()

    def _document_path(self, content_sha256: str) -> Path:
        if _CONTENT_HASH.fullmatch(str(content_sha256)) is None:
            raise TeachingResourceReviewError("resource review content hash is invalid")
        return self.root / f"{content_sha256}.review.json"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_path = self.root / ".resource-reviews.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        with self._thread_lock:
            try:
                descriptor = os.open(lock_path, flags, 0o600)
            except OSError as exc:
                raise TeachingResourceReviewError(
                    "resource review lock is unavailable"
                ) from exc
            try:
                os.fchmod(descriptor, 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def _read_document(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise TeachingResourceReviewError("resource review document is unsafe")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise TeachingResourceReviewError(
                "resource review document cannot be read"
            ) from exc
        if not raw or len(raw) > MAX_REVIEW_STORE_BYTES:
            raise TeachingResourceReviewError("resource review document is invalid")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TeachingResourceReviewError(
                "resource review document JSON is invalid"
            ) from exc
        if not isinstance(document, Mapping):
            raise TeachingResourceReviewError("resource review document is invalid")
        material = dict(document)
        declared = material.pop("document_sha256", None)
        if (
            document.get("schema") != RESOURCE_REVIEW_STORE_SCHEMA
            or document.get("content_sha256") != path.name.removesuffix(".review.json")
            or not isinstance(document.get("revisions"), list)
            or not 1 <= len(document["revisions"]) <= MAX_REVIEW_REVISIONS
            or document.get("current_version") != len(document["revisions"])
            or not isinstance(declared, str)
            or not hmac_compare(declared, canonical_sha256(material))
        ):
            raise TeachingResourceReviewError(
                "resource review document integrity failed"
            )
        previous: str | None = None
        seen_keys: set[str] = set()
        for expected_version, revision in enumerate(document["revisions"], 1):
            if not isinstance(revision, Mapping):
                raise TeachingResourceReviewError("resource review revision is invalid")
            revision_material = dict(revision)
            revision_sha = revision_material.pop("revision_sha256", None)
            if (
                revision.get("schema") != RESOURCE_REVIEW_REVISION_SCHEMA
                or revision.get("version") != expected_version
                or revision.get("previous_revision_sha256") != previous
                or not isinstance(revision.get("idempotency_key_sha256"), str)
                or revision["idempotency_key_sha256"] in seen_keys
                or not isinstance(revision.get("request_sha256"), str)
                or not isinstance(revision.get("projection"), Mapping)
                or not isinstance(revision_sha, str)
                or not hmac_compare(revision_sha, canonical_sha256(revision_material))
            ):
                raise TeachingResourceReviewError("resource review revision is invalid")
            projection = validate_reviewed_resource_projection(revision["projection"])
            receipt = projection["review_projection"]["receipt"]
            if (
                receipt["review_version"] != expected_version
                or receipt["previous_review_sha256"] != previous
                or projection["content_sha256"] != document["content_sha256"]
            ):
                raise TeachingResourceReviewError(
                    "resource review revision binding is invalid"
                )
            seen_keys.add(str(revision["idempotency_key_sha256"]))
            previous = str(revision_sha)
        return _json_copy(document)

    def _write_document(self, path: Path, document: Mapping[str, Any]) -> None:
        material = _json_copy(dict(document))
        material.pop("document_sha256", None)
        payload = {**material, "document_sha256": canonical_sha256(material)}
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_REVIEW_STORE_BYTES:
            raise TeachingResourceReviewError("resource review history is full")
        temporary_name = ""
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.root, prefix=".resource-review-", suffix=".tmp"
            )
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            temporary_name = ""
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise TeachingResourceReviewError(
                "resource review atomic write failed"
            ) from exc
        finally:
            if temporary_name:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def review(
        self,
        original_resource: Mapping[str, Any],
        request: Mapping[str, Any],
        authority_receipt: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        original = _descriptor_material(original_resource)
        validated_request = _validated_review_request(request, original=original)
        authority = _validated_authority_receipt(authority_receipt)
        content_hash = str(original["content_sha256"])
        path = self._document_path(content_hash)
        request_hash = canonical_sha256(validated_request)
        idempotency_hash = _text_sha256(
            str(validated_request["resource_review_idempotency_key"])
        )
        with self._locked():
            document = self._read_document(path)
            revisions = [] if document is None else list(document["revisions"])
            for revision in revisions:
                if revision["idempotency_key_sha256"] != idempotency_hash:
                    continue
                if revision["request_sha256"] != request_hash:
                    raise TeachingResourceReviewError(
                        "resource review idempotency key was reused"
                    )
                return _json_copy(revision["projection"])
            current_version = len(revisions)
            if validated_request["expected_review_version"] != current_version:
                raise TeachingResourceReviewError("resource review version conflict")
            if current_version >= MAX_REVIEW_REVISIONS:
                raise TeachingResourceReviewError(
                    "resource review revision limit reached"
                )
            previous_revision = (
                str(revisions[-1]["revision_sha256"]) if revisions else None
            )
            projection = build_reviewed_resource_projection(
                original,
                validated_request,
                authority,
                review_version=current_version + 1,
                previous_review_sha256=previous_revision,
                now=now,
            )
            revision_material = {
                "schema": RESOURCE_REVIEW_REVISION_SCHEMA,
                "version": current_version + 1,
                "previous_revision_sha256": previous_revision,
                "idempotency_key_sha256": idempotency_hash,
                "request_sha256": request_hash,
                "authority_receipt_sha256": authority["receipt_sha256"],
                "projection": projection,
            }
            revision = {
                **revision_material,
                "revision_sha256": canonical_sha256(revision_material),
            }
            revisions.append(revision)
            new_document = {
                "schema": RESOURCE_REVIEW_STORE_SCHEMA,
                "content_sha256": content_hash,
                "resource_id": original["resource_id"],
                "current_version": len(revisions),
                "revisions": revisions,
            }
            self._write_document(path, new_document)
            return _json_copy(projection)

    def get(self, content_sha256: str) -> dict[str, Any] | None:
        path = self._document_path(content_sha256)
        with self._locked():
            document = self._read_document(path)
            if document is None:
                return None
            return _json_copy(document["revisions"][-1]["projection"])

    def get_document(self, content_sha256: str) -> dict[str, Any] | None:
        """Return the validated append-only history for private export only."""

        path = self._document_path(content_sha256)
        with self._locked():
            document = self._read_document(path)
            return _json_copy(document) if document is not None else None

    def document_path(self, content_sha256: str) -> Path:
        """Resolve one validated hash-bound document path for ownership-aware purge."""

        return self._document_path(content_sha256)

    def get_by_resource_id(self, resource_id: str) -> dict[str, Any] | None:
        if _RESOURCE_ID.fullmatch(str(resource_id)) is None:
            raise TeachingResourceReviewError("resource review resource ID is invalid")
        prefix = str(resource_id).removeprefix("res_")
        with self._locked():
            matches = sorted(self.root.glob(f"{prefix}[0-9a-f]*.review.json"))
            if len(matches) > 1:
                raise TeachingResourceReviewError("resource review ID is ambiguous")
            if not matches:
                return None
            document = self._read_document(matches[0])
            if document is None or document.get("resource_id") != resource_id:
                raise TeachingResourceReviewError("resource review ID binding failed")
            return _json_copy(document["revisions"][-1]["projection"])


__all__ = [
    "MAX_REVIEW_REVISIONS",
    "RESOURCE_REVIEW_PROJECTION_SCHEMA",
    "RESOURCE_REVIEW_RECEIPT_SCHEMA",
    "RESOURCE_REVIEW_REVISION_SCHEMA",
    "RESOURCE_REVIEW_STORE_SCHEMA",
    "TeachingResourceReviewError",
    "TeachingResourceReviewStore",
    "build_reviewed_resource_projection",
    "resource_descriptor_sha256",
    "validate_reviewed_resource_projection",
]
