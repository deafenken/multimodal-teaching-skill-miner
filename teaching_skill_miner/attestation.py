"""Ed25519 registration and one-time consumption for external lockbox evaluation."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .io_utils import ensure_private_directory, ensure_private_file, read_json, write_json


REGISTRATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AttestationError(ValueError):
    """Raised when an external freeze registration cannot be trusted."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AttestationError("attestation payload is not canonical JSON") from exc
    return payload.encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AttestationError("registered_at_utc must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AttestationError("registered_at_utc is invalid") from exc
    if parsed > datetime.now(timezone.utc):
        raise AttestationError("freeze registration timestamp is in the future")
    return parsed


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise AttestationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def build_freeze_registration_request(
    *,
    registration_id: str,
    model_fingerprint: str,
    claim_contract_fingerprint: str,
    training_dataset_fingerprint: str,
    training_feature_bundle_fingerprint: str,
    external_dataset_fingerprint: str,
    external_feature_bundle_fingerprint: str,
    external_coverage_evidence_fingerprint: str,
    registered_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the exact payload an independent lockbox custodian must sign."""

    if not isinstance(registration_id, str) or not REGISTRATION_ID_RE.fullmatch(
        registration_id
    ):
        raise AttestationError(
            "registration_id must be 6-128 safe identifier characters"
        )
    timestamp = registered_at_utc or _utc_now()
    _parse_utc(timestamp)
    request = {
        "schema_version": "1.0",
        "protocol": "frozen_external_lockbox_registration_v1",
        "registration_id": registration_id,
        "registered_at_utc": timestamp,
        "model_fingerprint": _require_sha256(
            "model_fingerprint", model_fingerprint
        ),
        "claim_contract_fingerprint": _require_sha256(
            "claim_contract_fingerprint", claim_contract_fingerprint
        ),
        "training_dataset_fingerprint": _require_sha256(
            "training_dataset_fingerprint", training_dataset_fingerprint
        ),
        "training_feature_bundle_fingerprint": _require_sha256(
            "training_feature_bundle_fingerprint",
            training_feature_bundle_fingerprint,
        ),
        "external_dataset_fingerprint": _require_sha256(
            "external_dataset_fingerprint", external_dataset_fingerprint
        ),
        "external_feature_bundle_fingerprint": _require_sha256(
            "external_feature_bundle_fingerprint",
            external_feature_bundle_fingerprint,
        ),
        "external_coverage_evidence_fingerprint": _require_sha256(
            "external_coverage_evidence_fingerprint",
            external_coverage_evidence_fingerprint,
        ),
        "custodian_attestations": {
            "held_out_from_model_development": True,
            "deployment_target_documented_before_evaluation": True,
            "prospective_collection": True,
            "labels_not_used_to_modify_model_or_claim_contract": True,
            "maximum_evaluations": 1,
        },
    }
    request["request_fingerprint"] = hashlib.sha256(
        _canonical_bytes(request)
    ).hexdigest()
    return request


def sign_freeze_registration(
    request: Mapping[str, Any],
    *,
    private_key_pem: bytes,
    issuer: str,
    key_id: str,
) -> dict[str, Any]:
    """Sign a registration request with an Ed25519 custodian key."""

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 attestation requires `pip install -e '.[recognition]'`"
        ) from exc
    if not isinstance(issuer, str) or not issuer.strip():
        raise AttestationError("attestation issuer is required")
    if not isinstance(key_id, str) or not REGISTRATION_ID_RE.fullmatch(key_id):
        raise AttestationError("key_id must use safe identifier characters")
    validated = _validate_request(dict(request))
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise AttestationError("private key must be Ed25519")
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signature = private_key.sign(_canonical_bytes(validated))
    return {
        "schema_version": "1.0",
        "protocol": "ed25519_frozen_external_lockbox_attestation_v1",
        "issuer": issuer.strip(),
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
        "request": validated,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    fingerprint = request.pop("request_fingerprint", None)
    if (
        request.get("schema_version") != "1.0"
        or request.get("protocol") != "frozen_external_lockbox_registration_v1"
    ):
        raise AttestationError("unsupported freeze registration request")
    registration_id = request.get("registration_id")
    if not isinstance(registration_id, str) or not REGISTRATION_ID_RE.fullmatch(
        registration_id
    ):
        raise AttestationError("invalid freeze registration_id")
    _parse_utc(request.get("registered_at_utc"))
    for field in (
        "model_fingerprint",
        "claim_contract_fingerprint",
        "training_dataset_fingerprint",
        "training_feature_bundle_fingerprint",
        "external_dataset_fingerprint",
        "external_feature_bundle_fingerprint",
        "external_coverage_evidence_fingerprint",
    ):
        _require_sha256(field, request.get(field))
    conditions = request.get("custodian_attestations")
    expected_conditions = {
        "held_out_from_model_development": True,
        "deployment_target_documented_before_evaluation": True,
        "prospective_collection": True,
        "labels_not_used_to_modify_model_or_claim_contract": True,
        "maximum_evaluations": 1,
    }
    if conditions != expected_conditions:
        raise AttestationError("custodian attestations are incomplete or weakened")
    expected_fingerprint = hashlib.sha256(_canonical_bytes(request)).hexdigest()
    if fingerprint != expected_fingerprint:
        raise AttestationError("freeze registration request fingerprint mismatch")
    return {**request, "request_fingerprint": expected_fingerprint}


def verify_freeze_registration(
    attestation: Mapping[str, Any],
    *,
    trusted_public_key_pem: bytes,
    expected_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Verify the custodian signature and every model/dataset binding."""

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 attestation requires `pip install -e '.[recognition]'`"
        ) from exc
    if (
        attestation.get("schema_version") != "1.0"
        or attestation.get("protocol")
        != "ed25519_frozen_external_lockbox_attestation_v1"
    ):
        raise AttestationError("unsupported freeze registration attestation")
    if not isinstance(attestation.get("issuer"), str) or not attestation["issuer"].strip():
        raise AttestationError("attestation issuer is missing")
    request_raw = attestation.get("request")
    if not isinstance(request_raw, Mapping):
        raise AttestationError("attestation request is missing")
    request = _validate_request(dict(request_raw))
    public_key = serialization.load_pem_public_key(trusted_public_key_pem)
    if not isinstance(public_key, Ed25519PublicKey):
        raise AttestationError("trusted public key must be Ed25519")
    public_der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_fingerprint = hashlib.sha256(public_der).hexdigest()
    if attestation.get("public_key_sha256") != public_fingerprint:
        raise AttestationError("attestation public-key fingerprint mismatch")
    try:
        signature = base64.b64decode(
            str(attestation.get("signature_base64", "")), validate=True
        )
        public_key.verify(signature, _canonical_bytes(request))
    except (ValueError, InvalidSignature) as exc:
        raise AttestationError("freeze registration signature verification failed") from exc
    for name, expected in expected_bindings.items():
        if request.get(name) != expected:
            raise AttestationError(f"freeze registration binding mismatch: {name}")
    canonical_attestation = dict(attestation)
    return {
        "verified": True,
        "registration_id": request["registration_id"],
        "registered_at_utc": request["registered_at_utc"],
        "issuer": attestation["issuer"],
        "key_id": attestation.get("key_id"),
        "public_key_sha256": public_fingerprint,
        "request_fingerprint": request["request_fingerprint"],
        "attestation_fingerprint": hashlib.sha256(
            _canonical_bytes(canonical_attestation)
        ).hexdigest(),
        "custodian_attestations": request["custodian_attestations"],
    }


def consume_registration_once(
    ledger_dir: str | Path,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically reserve a signed registration ID so repeat evaluation fails closed."""

    if verification.get("verified") is not True:
        raise AttestationError("only a verified registration can be consumed")
    registration_id = verification.get("registration_id")
    if not isinstance(registration_id, str) or not REGISTRATION_ID_RE.fullmatch(
        registration_id
    ):
        raise AttestationError("invalid verified registration_id")
    ledger = ensure_private_directory(ledger_dir)
    receipt_path = ledger / f"{registration_id}.json"
    receipt = {
        "schema_version": "1.0",
        "protocol": "one_time_lockbox_consumption_receipt_v1",
        "registration_id": registration_id,
        "attestation_fingerprint": verification["attestation_fingerprint"],
        "consumed_at_utc": _utc_now(),
        "status": "reserved_before_metric_computation",
    }
    payload = json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(
            receipt_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise AttestationError(
            f"freeze registration was already consumed: {registration_id}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    ensure_private_file(receipt_path)
    return {
        **receipt,
        "receipt_fingerprint": hashlib.sha256(payload).hexdigest(),
        "ledger_path": str(receipt_path),
    }


def finalize_consumption_receipt(
    receipt: Mapping[str, Any],
    *,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    """Bind the reserved one-time receipt to the completed evaluation."""

    path = Path(str(receipt.get("ledger_path", "")))
    if not path.is_file():
        raise AttestationError("reserved one-time receipt is missing")
    current = read_json(path)
    if current.get("status") != "reserved_before_metric_computation":
        raise AttestationError("one-time receipt is not in the reserved state")
    _require_sha256("evaluation_fingerprint", evaluation_fingerprint)
    final = {
        **current,
        "status": "completed",
        "evaluation_fingerprint": evaluation_fingerprint,
        "completed_at_utc": _utc_now(),
    }
    write_json(path, final)
    payload = path.read_bytes()
    return {
        **final,
        "receipt_fingerprint": hashlib.sha256(payload).hexdigest(),
    }


def sign_evaluation_report_receipt(
    *,
    evaluation_report_fingerprint: str,
    registration_id: str,
    registration_attestation_fingerprint: str,
    deployment_accuracy_established: bool,
    private_key_pem: bytes,
    issuer: str,
    key_id: str,
) -> dict[str, Any]:
    """Sign the completed evaluation report for transport and delivery audit."""

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 attestation requires `pip install -e '.[recognition]'`"
        ) from exc
    _require_sha256("evaluation_report_fingerprint", evaluation_report_fingerprint)
    _require_sha256(
        "registration_attestation_fingerprint",
        registration_attestation_fingerprint,
    )
    if not REGISTRATION_ID_RE.fullmatch(registration_id):
        raise AttestationError("invalid evaluation registration_id")
    if not isinstance(deployment_accuracy_established, bool):
        raise AttestationError("deployment accuracy status must be boolean")
    if not issuer.strip() or not REGISTRATION_ID_RE.fullmatch(key_id):
        raise AttestationError("issuer and key_id are required")
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise AttestationError("private key must be Ed25519")
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    payload = {
        "schema_version": "1.0",
        "protocol": "external_evaluation_report_receipt_v1",
        "registration_id": registration_id,
        "registration_attestation_fingerprint": registration_attestation_fingerprint,
        "evaluation_report_fingerprint": evaluation_report_fingerprint,
        "deployment_accuracy_established": deployment_accuracy_established,
        "signed_at_utc": _utc_now(),
    }
    signature = private_key.sign(_canonical_bytes(payload))
    return {
        "schema_version": "1.0",
        "protocol": "ed25519_external_evaluation_report_receipt_v1",
        "issuer": issuer.strip(),
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
        "payload": payload,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_evaluation_report_receipt(
    receipt: Mapping[str, Any],
    *,
    trusted_public_key_pem: bytes,
    expected_report_fingerprint: str,
    expected_registration_id: str,
    expected_registration_attestation_fingerprint: str,
) -> dict[str, Any]:
    """Verify a custodian-signed completed evaluation report receipt."""

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 attestation requires `pip install -e '.[recognition]'`"
        ) from exc
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("protocol")
        != "ed25519_external_evaluation_report_receipt_v1"
    ):
        raise AttestationError("unsupported external evaluation receipt")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        raise AttestationError("external evaluation receipt payload is missing")
    expected_payload = {
        "schema_version": "1.0",
        "protocol": "external_evaluation_report_receipt_v1",
        "registration_id": expected_registration_id,
        "registration_attestation_fingerprint": (
            expected_registration_attestation_fingerprint
        ),
        "evaluation_report_fingerprint": expected_report_fingerprint,
        "deployment_accuracy_established": payload.get(
            "deployment_accuracy_established"
        ),
        "signed_at_utc": payload.get("signed_at_utc"),
    }
    if dict(payload) != expected_payload:
        raise AttestationError("external evaluation receipt binding mismatch")
    _parse_utc(payload.get("signed_at_utc"))
    if not isinstance(payload.get("deployment_accuracy_established"), bool):
        raise AttestationError("external evaluation receipt claim is malformed")
    public_key = serialization.load_pem_public_key(trusted_public_key_pem)
    if not isinstance(public_key, Ed25519PublicKey):
        raise AttestationError("trusted public key must be Ed25519")
    public_der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_fingerprint = hashlib.sha256(public_der).hexdigest()
    if receipt.get("public_key_sha256") != public_fingerprint:
        raise AttestationError("external evaluation receipt public-key mismatch")
    try:
        signature = base64.b64decode(
            str(receipt.get("signature_base64", "")), validate=True
        )
        public_key.verify(signature, _canonical_bytes(payload))
    except (ValueError, InvalidSignature) as exc:
        raise AttestationError("external evaluation receipt signature failed") from exc
    return {
        "verified": True,
        "issuer": receipt.get("issuer"),
        "key_id": receipt.get("key_id"),
        "public_key_sha256": public_fingerprint,
        "deployment_accuracy_established": payload[
            "deployment_accuracy_established"
        ],
        "receipt_fingerprint": hashlib.sha256(_canonical_bytes(dict(receipt))).hexdigest(),
    }
