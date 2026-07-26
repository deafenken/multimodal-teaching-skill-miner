"""Strict evaluation primitives for defensible classroom-recognition claims.

This module deliberately does not infer session, participant, teacher, cohort,
or site identities from filenames.  A dataset importer must attest the identity
fields required by its task and provide a content fingerprint for every
synchronized example.  The evaluators fail closed when that evidence is absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..attestation import (
    consume_registration_once,
    finalize_consumption_receipt,
    verify_freeze_registration,
)
from .metrics import classification_metrics


IDENTITY_FIELDS = ("session_id", "teacher_id", "site_id")
GROUP_IDENTITY_FIELDS = (
    "session_id",
    "participant_id",
    "teacher_id",
    "cohort_id",
    "activity_id",
    "site_id",
)
REQUIRED_AUDIT_FLAGS = (
    "provenance_verified",
    "real_classroom_recording",
    "independent_human_ground_truth",
    "synchronized_modalities_verified",
)
GROUPED_ACCURACY_THRESHOLDS = {
    "minimum_accuracy": 0.75,
    "minimum_macro_f1": 0.60,
    "minimum_accuracy_ci_lower": 0.65,
    "minimum_macro_f1_ci_lower": 0.50,
    "minimum_per_class_recall": 0.50,
    "minimum_per_class_support": 10,
}


class CredibilityError(ValueError):
    """Raised when an input cannot support a strict accuracy claim."""


@dataclass(frozen=True)
class ClaimContract:
    """Thresholds and collection conditions frozen before external evaluation.

    The contract is serialized inside the model checkpoint, so an evaluator
    cannot weaken a performance, coverage, or lockbox requirement after seeing
    external labels.  Defaults are intentionally conservative but remain
    suitable for the small, perfectly separable fixtures used by this module's
    tests.
    """

    primary_metric: str = "macro_f1"
    minimum_primary_metric: float = 0.60
    minimum_accuracy: float = 0.75
    minimum_macro_f1: float = 0.60
    minimum_accuracy_ci_lower: float = 0.65
    minimum_macro_f1_ci_lower: float = 0.50
    minimum_per_class_recall: float = 0.50
    minimum_per_class_support: int = 10
    minimum_sessions: int = 10
    minimum_coverage_fraction: float = 0.80
    require_prospective: bool = True
    require_one_time_lockbox: bool = True


def _claim_contract_payload(contract: ClaimContract) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "primary_metric": contract.primary_metric,
        "minimum_primary_metric": contract.minimum_primary_metric,
        "minimum_accuracy": contract.minimum_accuracy,
        "minimum_macro_f1": contract.minimum_macro_f1,
        "minimum_accuracy_ci_lower": contract.minimum_accuracy_ci_lower,
        "minimum_macro_f1_ci_lower": contract.minimum_macro_f1_ci_lower,
        "minimum_per_class_recall": contract.minimum_per_class_recall,
        "minimum_per_class_support": contract.minimum_per_class_support,
        "minimum_sessions": contract.minimum_sessions,
        "minimum_coverage_fraction": contract.minimum_coverage_fraction,
        "require_prospective": contract.require_prospective,
        "require_one_time_lockbox": contract.require_one_time_lockbox,
    }


def _normalize_claim_contract(
    value: ClaimContract | Mapping[str, Any] | None,
) -> ClaimContract:
    if value is None:
        contract = ClaimContract()
    elif isinstance(value, ClaimContract):
        contract = value
    elif isinstance(value, Mapping):
        raw = dict(value)
        schema_version = raw.pop("schema_version", "1.0")
        if schema_version != "1.0":
            raise CredibilityError("unsupported claim contract schema")
        try:
            contract = ClaimContract(**raw)
        except TypeError as exc:
            raise CredibilityError("malformed claim contract") from exc
    else:
        raise CredibilityError("claim_contract must be a ClaimContract or object")

    if contract.primary_metric not in {"accuracy", "macro_f1"}:
        raise CredibilityError("claim contract primary_metric must be accuracy or macro_f1")
    thresholds = {
        "minimum_primary_metric": contract.minimum_primary_metric,
        "minimum_accuracy": contract.minimum_accuracy,
        "minimum_macro_f1": contract.minimum_macro_f1,
        "minimum_accuracy_ci_lower": contract.minimum_accuracy_ci_lower,
        "minimum_macro_f1_ci_lower": contract.minimum_macro_f1_ci_lower,
        "minimum_per_class_recall": contract.minimum_per_class_recall,
        "minimum_coverage_fraction": contract.minimum_coverage_fraction,
    }
    for name, threshold in thresholds.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise CredibilityError(f"claim contract {name} must be numeric")
        if not 0.0 <= float(threshold) <= 1.0:
            raise CredibilityError(f"claim contract {name} must be between zero and one")
    if (
        isinstance(contract.minimum_per_class_support, bool)
        or not isinstance(contract.minimum_per_class_support, int)
        or contract.minimum_per_class_support < 1
    ):
        raise CredibilityError("claim contract minimum_per_class_support must be positive")
    if (
        isinstance(contract.minimum_sessions, bool)
        or not isinstance(contract.minimum_sessions, int)
        or contract.minimum_sessions < 2
    ):
        raise CredibilityError("claim contract minimum_sessions must be at least two")
    if not isinstance(contract.require_prospective, bool) or not isinstance(
        contract.require_one_time_lockbox, bool
    ):
        raise CredibilityError("claim contract requirement flags must be booleans")
    return contract


def _dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("install optional dependencies with `pip install -e '.[recognition]'`") from exc
    return np, LogisticRegression, StratifiedGroupKFold, StandardScaler, (accuracy_score, f1_score)


def strict_dataset_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the label, verified identities, modalities, and source bytes."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise CredibilityError("strict dataset fingerprint requires non-empty records")
    canonical = []
    observed_sample_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CredibilityError(f"record {index} is not an object")
        sample_id = record.get("sample_id")
        content_sha256 = record.get("content_sha256")
        label = record.get("label")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise CredibilityError(f"record {index} lacks a string sample_id")
        if sample_id in observed_sample_ids:
            raise CredibilityError(f"duplicate sample_id in strict dataset: {sample_id}")
        observed_sample_ids.add(sample_id)
        if (
            not isinstance(content_sha256, str)
            or len(content_sha256) != 64
            or any(char not in "0123456789abcdef" for char in content_sha256.lower())
        ):
            raise CredibilityError(f"record {sample_id} lacks a valid content_sha256")
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise CredibilityError(f"record {sample_id} lacks a non-negative integer label")
        modalities = record.get("modalities", {})
        if not isinstance(modalities, Mapping) or not modalities:
            raise CredibilityError(f"record {sample_id} lacks modality-presence metadata")
        normalized_modalities: list[tuple[str, bool]] = []
        for name, present in modalities.items():
            if not isinstance(name, str) or not name:
                raise CredibilityError(f"record {sample_id} has an invalid modality name")
            if not isinstance(present, bool):
                raise CredibilityError(f"record {sample_id} has a non-boolean modality flag")
            normalized_modalities.append((name, present))
        normalized_identities: list[tuple[str, str]] = []
        for name, value in record.items():
            if str(name).endswith("_id"):
                if not isinstance(name, str) or not isinstance(value, str) or not value.strip():
                    raise CredibilityError(
                        f"record {sample_id} has an invalid identity field {name}"
                    )
                normalized_identities.append((name, value))
        canonical.append(
            {
                "sample_id": sample_id,
                "content_sha256": content_sha256.lower(),
                "label": label,
                "identities": sorted(normalized_identities),
                "modalities": sorted(normalized_modalities),
            }
        )
    canonical.sort(key=lambda item: str(item["sample_id"]))
    payload = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strict_feature_bundle_fingerprint(
    records: Sequence[Mapping[str, Any]],
    feature_names_by_modality: Mapping[str, Sequence[str]],
    modality_features: Mapping[str, Sequence[Sequence[float]]],
    feature_provenance: Mapping[str, Mapping[str, Any]],
) -> str:
    """Bind generic ordered feature matrices, schemas, and extractor provenance."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise CredibilityError("strict feature bundle requires non-empty records")
    if not isinstance(feature_names_by_modality, Mapping) or not isinstance(
        modality_features, Mapping
    ) or not isinstance(feature_provenance, Mapping):
        raise CredibilityError("strict feature bundle components must be objects")
    modalities = sorted(str(name) for name in modality_features)
    schema_modalities = {str(name) for name in feature_names_by_modality}
    provenance_modalities = {str(name) for name in feature_provenance}
    if (
        not modalities
        or len(modalities) != len(modality_features)
        or set(modalities) != schema_modalities
        or set(modalities) != provenance_modalities
    ):
        raise CredibilityError("strict feature bundle modality schemas are inconsistent")
    if any(not isinstance(name, str) or not name for name in modality_features):
        raise CredibilityError("strict feature bundle modality names must be non-empty strings")
    normalized_provenance = validate_feature_provenance(feature_provenance, modalities)
    digest = hashlib.sha256()

    def add_text(value: Any) -> None:
        payload = str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    add_text("strict_feature_bundle_v1")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CredibilityError(f"record {index} is not an object")
        for field in ("sample_id", "content_sha256", "label"):
            if field not in record:
                raise CredibilityError(f"record {index} lacks {field}")
            add_text(record[field])
    for modality in modalities:
        raw_names = feature_names_by_modality[modality]
        if (
            isinstance(raw_names, (str, bytes, Mapping))
            or not hasattr(raw_names, "__len__")
            or not hasattr(raw_names, "__iter__")
        ):
            raise CredibilityError(f"invalid feature names for modality {modality}")
        names = list(raw_names)
        matrix = modality_features[modality]
        if (
            not names
            or any(not isinstance(value, str) or not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise CredibilityError(f"invalid feature names for modality {modality}")
        if (
            isinstance(matrix, (str, bytes, Mapping))
            or not hasattr(matrix, "__len__")
            or not hasattr(matrix, "__iter__")
        ):
            raise CredibilityError(f"invalid feature matrix for modality {modality}")
        if len(matrix) != len(records):
            raise CredibilityError(f"feature row count mismatch for modality {modality}")
        add_text(modality)
        add_text(
            json.dumps(
                normalized_provenance[modality],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for name in names:
            add_text(name)
        add_text(len(matrix))
        add_text(len(names))
        for row_index, row in enumerate(matrix):
            if (
                isinstance(row, (str, bytes, Mapping))
                or not hasattr(row, "__len__")
                or not hasattr(row, "__iter__")
            ):
                raise CredibilityError(
                    f"invalid feature row in {modality} row {row_index}"
                )
            if len(row) != len(names):
                raise CredibilityError(
                    f"feature width mismatch for {modality} row {row_index}"
                )
            for value in row:
                if isinstance(value, bool):
                    raise CredibilityError(
                        f"non-numeric feature in {modality} row {row_index}"
                    )
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise CredibilityError(
                        f"non-numeric feature in {modality} row {row_index}"
                    ) from exc
                if not math.isfinite(numeric):
                    raise CredibilityError(
                        f"non-finite feature in {modality} row {row_index}"
                    )
                digest.update(struct.pack(">d", numeric))
    return digest.hexdigest()


def validate_strict_manifest(
    manifest: Mapping[str, Any],
    *,
    required_modalities: Sequence[str] = (),
    required_identity_fields: Sequence[str] = IDENTITY_FIELDS,
) -> dict[str, Any]:
    """Validate evidence needed for strict cross-group or external evaluation.

    Verification is explicit: guessed filename groups and undocumented identity
    columns are rejected even when their values happen to look plausible.
    """

    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "2.0":
        raise CredibilityError("strict manifest schema_version must be 2.0")
    audit = manifest.get("audit")
    records = manifest.get("records")
    if not isinstance(audit, Mapping) or not isinstance(records, list) or not records:
        raise CredibilityError("strict manifest requires a non-empty records list and audit object")
    if not isinstance(audit.get("dataset_id"), str) or not audit["dataset_id"].strip():
        raise CredibilityError("audit.dataset_id must be a non-empty string")

    missing_flags = [name for name in REQUIRED_AUDIT_FLAGS if audit.get(name) is not True]
    if missing_flags:
        raise CredibilityError(
            "strict manifest requires verified audit flags: " + ", ".join(missing_flags)
        )
    identity_verification = audit.get("identity_metadata_verified")
    if not isinstance(identity_verification, Mapping):
        raise CredibilityError("audit.identity_metadata_verified must document session/teacher/site IDs")
    if any(not isinstance(name, str) for name in required_identity_fields):
        raise CredibilityError("required identity fields must be strings")
    normalized_identity_fields = tuple(dict.fromkeys(required_identity_fields))
    invalid_identity_fields = [
        name for name in normalized_identity_fields if name not in GROUP_IDENTITY_FIELDS
    ]
    if invalid_identity_fields:
        raise CredibilityError(
            "unsupported identity metadata fields: " + ", ".join(invalid_identity_fields)
        )
    unverified_ids = [
        name for name in normalized_identity_fields if identity_verification.get(name) is not True
    ]
    if unverified_ids:
        raise CredibilityError("identity metadata is not verified: " + ", ".join(unverified_ids))
    for documentation_field in ("identity_metadata_source", "ground_truth_source"):
        if not isinstance(audit.get(documentation_field), str) or not audit[
            documentation_field
        ].strip():
            raise CredibilityError(f"audit.{documentation_field} must document evidence provenance")

    sample_ids: set[str] = set()
    content_hashes: set[str] = set()
    session_to_site: dict[str, str] = {}
    identity_values: dict[str, set[str]] = {
        name: set() for name in normalized_identity_fields
    }
    labels: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CredibilityError(f"record {index} is not an object")
        raw_sample_id = record.get("sample_id")
        sample_id = raw_sample_id.strip() if isinstance(raw_sample_id, str) else ""
        if not sample_id or sample_id in sample_ids:
            raise CredibilityError(f"record {index} has a missing or duplicate sample_id")
        sample_ids.add(sample_id)
        raw_content_hash = record.get("content_sha256")
        content_hash = raw_content_hash.lower() if isinstance(raw_content_hash, str) else ""
        if len(content_hash) != 64 or any(char not in "0123456789abcdef" for char in content_hash):
            raise CredibilityError(f"record {sample_id} lacks a valid content_sha256")
        if content_hash in content_hashes:
            raise CredibilityError(f"duplicate synchronized content detected at record {sample_id}")
        content_hashes.add(content_hash)

        try:
            raw_label = record["label"]
            if isinstance(raw_label, bool) or not isinstance(raw_label, int):
                raise TypeError
            label = raw_label
        except (KeyError, TypeError, ValueError) as exc:
            raise CredibilityError(f"record {sample_id} lacks an integer label") from exc
        if label < 0:
            raise CredibilityError(f"record {sample_id} has a negative label")
        labels.add(label)

        identities: dict[str, str] = {}
        for field_name in normalized_identity_fields:
            raw_value = record.get(field_name)
            value = raw_value.strip() if isinstance(raw_value, str) else ""
            if not value:
                raise CredibilityError(f"record {sample_id} lacks verified {field_name}")
            identities[field_name] = value
            identity_values[field_name].add(value)
        if "session_id" in identities and "site_id" in identities:
            session_id = identities["session_id"]
            site_id = identities["site_id"]
            if session_id in session_to_site and session_to_site[session_id] != site_id:
                raise CredibilityError(f"session {session_id} maps to multiple sites")
            session_to_site[session_id] = site_id

        modalities = record.get("modalities")
        if not isinstance(modalities, Mapping):
            raise CredibilityError(f"record {sample_id} lacks modality-presence metadata")
        missing_modalities = [name for name in required_modalities if modalities.get(name) is not True]
        if missing_modalities:
            raise CredibilityError(
                f"record {sample_id} lacks synchronized modalities: {', '.join(missing_modalities)}"
            )

    expected_fingerprint = strict_dataset_fingerprint(records)
    if audit.get("dataset_fingerprint") != expected_fingerprint:
        raise CredibilityError("audit.dataset_fingerprint does not match strict manifest contents")
    if labels != set(range(max(labels) + 1)):
        raise CredibilityError("labels must be contiguous integers starting at zero")

    return {
        "sample_count": len(records),
        "class_count": len(labels),
        **{
            f"{name.removesuffix('_id')}_count": len(values)
            for name, values in identity_values.items()
        },
        "dataset_fingerprint": expected_fingerprint,
        "required_modalities": list(required_modalities),
        "required_identity_fields": list(normalized_identity_fields),
        "strict_metadata_verified": True,
    }


def validate_feature_provenance(
    feature_provenance: Mapping[str, Mapping[str, Any]],
    modalities: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Require fixed, label-blind extractors for precomputed feature matrices."""

    normalized: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        evidence = feature_provenance.get(modality)
        if not isinstance(evidence, Mapping):
            raise CredibilityError(f"missing feature provenance for modality {modality}")
        raw_extractor_id = evidence.get("extractor_id")
        raw_fingerprint = evidence.get("extractor_fingerprint")
        if not isinstance(raw_extractor_id, str) or not raw_extractor_id.strip():
            raise CredibilityError(f"feature extractor ID is missing for modality {modality}")
        if not isinstance(raw_fingerprint, str):
            raise CredibilityError(f"feature extractor fingerprint is invalid for modality {modality}")
        extractor_id = raw_extractor_id.strip()
        fingerprint = raw_fingerprint.lower()
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise CredibilityError(f"feature extractor fingerprint is invalid for modality {modality}")
        required_state = {
            "frozen_before_evaluation": True,
            "uses_ground_truth_labels": False,
            "fitted_on_evaluation_records": False,
        }
        for field_name, required_value in required_state.items():
            if evidence.get(field_name) is not required_value:
                raise CredibilityError(
                    f"feature provenance for {modality} violates {field_name}={required_value}"
                )
        try:
            extra = {
                str(name): value
                for name, value in evidence.items()
                if name not in {"extractor_id", "extractor_fingerprint", *required_state}
            }
            json.dumps(extra, ensure_ascii=True, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise CredibilityError(
                f"feature provenance for {modality} is not canonical JSON"
            ) from exc
        normalized[modality] = {
            "extractor_id": extractor_id,
            "extractor_fingerprint": fingerprint,
            **required_state,
            **extra,
        }
    return normalized


def validate_class_schema(
    class_names: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_class_count: int,
) -> tuple[str, ...]:
    """Bind integer labels to unique, non-empty semantic class names."""

    if not isinstance(class_names, Sequence) or isinstance(class_names, (str, bytes)):
        raise CredibilityError("class_names must be an ordered string sequence")
    normalized = tuple(class_names)
    if (
        len(normalized) != expected_class_count
        or any(not isinstance(value, str) or not value.strip() for value in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise CredibilityError(
            "class_names must contain one unique non-empty string per integer label"
        )
    normalized = tuple(value.strip() for value in normalized)
    observed: dict[int, set[str]] = {index: set() for index in range(expected_class_count)}
    for record in records:
        label = int(record["label"])
        label_name = record.get("label_name")
        if not isinstance(label_name, str) or not label_name.strip():
            raise CredibilityError(
                "strict accuracy claims require label_name on every manifest record"
            )
        observed[label].add(label_name.strip())
    mismatches = {
        label: sorted(values)
        for label, values in observed.items()
        if values != {normalized[label]}
    }
    if mismatches:
        raise CredibilityError(
            f"manifest label_name values do not match the ordered class schema: {mismatches}"
        )
    return normalized


def strict_coverage_evidence_fingerprint(evidence: Mapping[str, Any]) -> str:
    """Fingerprint the eligible denominator and complete-case exclusion accounting."""

    if not isinstance(evidence, Mapping):
        raise CredibilityError("evaluation coverage evidence must be an object")
    payload = {
        key: value
        for key, value in evidence.items()
        if key != "coverage_evidence_fingerprint"
    }
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CredibilityError("evaluation coverage evidence is not canonical JSON") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_evaluation_coverage(
    audit: Mapping[str, Any],
    *,
    evaluated_sample_count: int,
) -> dict[str, Any]:
    """Verify coverage fraction, denominator identity, and exclusion accounting."""

    evidence = audit.get("evaluation_coverage")
    if not isinstance(evidence, Mapping):
        raise CredibilityError(
            "external evaluation requires audit.evaluation_coverage evidence"
        )
    try:
        eligible = evidence["eligible_sample_count"]
        evaluated = evidence["evaluated_sample_count"]
        excluded = evidence["excluded_sample_count"]
    except KeyError as exc:
        raise CredibilityError("evaluation coverage counts are incomplete") from exc
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (eligible, evaluated, excluded)):
        raise CredibilityError("evaluation coverage counts must be integers")
    if eligible < 1 or evaluated != evaluated_sample_count or excluded != eligible - evaluated or excluded < 0:
        raise CredibilityError("evaluation coverage counts are inconsistent")
    denominator_fingerprint = evidence.get("eligible_population_fingerprint")
    if (
        not isinstance(denominator_fingerprint, str)
        or len(denominator_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in denominator_fingerprint)
    ):
        raise CredibilityError("eligible population fingerprint is invalid")
    if not isinstance(evidence.get("denominator_source"), str) or not evidence[
        "denominator_source"
    ].strip():
        raise CredibilityError("evaluation coverage denominator source is missing")
    reason_counts = evidence.get("exclusion_reason_counts")
    if not isinstance(reason_counts, Mapping) or any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in reason_counts.items()
    ):
        raise CredibilityError("evaluation exclusion reason counts are invalid")
    if sum(reason_counts.values()) != excluded:
        raise CredibilityError("evaluation exclusion reasons do not sum to excluded count")
    fraction = evaluated / eligible
    declared_fraction = audit.get("evaluation_coverage_fraction")
    if (
        isinstance(declared_fraction, bool)
        or not isinstance(declared_fraction, (int, float))
        or abs(float(declared_fraction) - fraction) > 1e-12
    ):
        raise CredibilityError("evaluation coverage fraction does not match its denominator")
    expected_fingerprint = strict_coverage_evidence_fingerprint(evidence)
    if evidence.get("coverage_evidence_fingerprint") != expected_fingerprint:
        raise CredibilityError("evaluation coverage evidence fingerprint mismatch")
    return {
        "eligible_sample_count": eligible,
        "evaluated_sample_count": evaluated,
        "excluded_sample_count": excluded,
        "coverage_fraction": fraction,
        "eligible_population_fingerprint": denominator_fingerprint,
        "denominator_source": evidence["denominator_source"],
        "exclusion_reason_counts": dict(reason_counts),
        "coverage_evidence_fingerprint": expected_fingerprint,
    }


def _prepare_feature_sets(
    modality_features: Mapping[str, Any],
    *,
    sample_count: int,
) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    np, _, _, _, _ = _dependencies()
    if len(modality_features) < 2:
        raise CredibilityError("multimodal evaluation requires at least two base modalities")
    if "fusion" in modality_features or "best_unimodal_nested" in modality_features:
        raise CredibilityError("fusion and best_unimodal_nested are reserved modality names")
    base_names = sorted(str(name) for name in modality_features)
    prepared: dict[str, Any] = {}
    dimensions: dict[str, int] = {}
    for name in base_names:
        matrix = np.asarray(modality_features[name], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != sample_count or matrix.shape[1] < 1:
            raise CredibilityError(
                f"{name} features must have shape ({sample_count}, positive_dimension)"
            )
        if not np.isfinite(matrix).all():
            raise CredibilityError(f"{name} features contain NaN or infinity")
        prepared[name] = matrix
        dimensions[name] = int(matrix.shape[1])
    prepared["fusion"] = np.concatenate([prepared[name] for name in base_names], axis=1)
    return prepared, base_names, dimensions


def _aligned_probabilities(model: Any, matrix: Any, class_count: int) -> Any:
    np, _, _, _, _ = _dependencies()
    raw = model.predict_proba(matrix)
    aligned = np.zeros((len(matrix), class_count), dtype=float)
    for column, label in enumerate(model.classes_):
        aligned[:, int(label)] = raw[:, column]
    return aligned


def _fit_predict(
    train_x: Any,
    train_y: Any,
    test_x: Any,
    *,
    c_value: float,
    class_count: int,
) -> tuple[Any, Any, Any]:
    _, LogisticRegression, _, StandardScaler, _ = _dependencies()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=2026,
    )
    model.fit(train_scaled, train_y)
    if set(int(value) for value in model.classes_) != set(range(class_count)):
        raise CredibilityError("a training fold is missing one or more classes")
    probabilities = _aligned_probabilities(model, scaler.transform(test_x), class_count)
    return probabilities, scaler, model


def _assert_class_group_support(labels: Any, groups: Any, *, splits: int, scope: str) -> None:
    np, _, _, _, _ = _dependencies()
    for label in np.unique(labels):
        group_count = len(np.unique(groups[labels == label]))
        if group_count < splits:
            raise CredibilityError(
                f"{scope}: class {int(label)} occurs in {group_count} groups; {splits} are required"
            )


def _select_c_nested(
    matrix: Any,
    labels: Any,
    groups: Any,
    *,
    class_count: int,
    inner_splits: int,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    np, _, StratifiedGroupKFold, _, metrics = _dependencies()
    _, f1_score = metrics
    _assert_class_group_support(labels, groups, splits=inner_splits, scope="inner CV")
    splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(matrix, labels, groups))
    candidates: list[dict[str, Any]] = []
    for c_value in c_grid:
        if float(c_value) <= 0:
            raise CredibilityError("all C values must be positive")
        probabilities = np.zeros((len(labels), class_count), dtype=float)
        assigned = np.zeros(len(labels), dtype=bool)
        fold_scores: list[float] = []
        for train_indices, validation_indices in splits:
            if set(groups[train_indices]) & set(groups[validation_indices]):
                raise RuntimeError("group leakage detected in inner CV")
            fold_probabilities, _, _ = _fit_predict(
                matrix[train_indices],
                labels[train_indices],
                matrix[validation_indices],
                c_value=c_value,
                class_count=class_count,
            )
            probabilities[validation_indices] = fold_probabilities
            assigned[validation_indices] = True
            fold_scores.append(
                float(
                    f1_score(
                        labels[validation_indices],
                        fold_probabilities.argmax(axis=1),
                        labels=list(range(class_count)),
                        average="macro",
                        zero_division=0,
                    )
                )
            )
        if not assigned.all():
            raise RuntimeError("inner CV did not assign every training example")
        score = float(
            f1_score(
                labels,
                probabilities.argmax(axis=1),
                labels=list(range(class_count)),
                average="macro",
                zero_division=0,
            )
        )
        candidates.append(
            {
                "C": float(c_value),
                "inner_oof_macro_f1": round(score, 6),
                "fold_macro_f1": [round(value, 6) for value in fold_scores],
            }
        )
    best = sorted(candidates, key=lambda item: (-item["inner_oof_macro_f1"], item["C"]))[0]
    return float(best["C"]), float(best["inner_oof_macro_f1"]), candidates


def _metric_value(y_true: Any, probabilities: Any, metric: str, class_count: int) -> float:
    _, _, _, _, metrics = _dependencies()
    accuracy_score, f1_score = metrics
    predictions = probabilities.argmax(axis=1)
    if metric == "accuracy":
        return float(accuracy_score(y_true, predictions))
    if metric == "macro_f1":
        return float(
            f1_score(
                y_true,
                predictions,
                labels=list(range(class_count)),
                average="macro",
                zero_division=0,
            )
        )
    raise ValueError(f"unsupported metric: {metric}")


def paired_group_comparison(
    y_true: Any,
    candidate_probabilities: Any,
    reference_probabilities: Any,
    groups: Any,
    *,
    class_count: int,
    bootstrap_replicates: int = 2000,
    permutation_replicates: int = 5000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Paired cluster bootstrap CIs and group-wise swap permutation tests."""

    np, _, _, _, _ = _dependencies()
    if bootstrap_replicates < 1 or permutation_replicates < 1:
        raise CredibilityError("bootstrap and permutation replicate counts must be positive")
    y_true = np.asarray(y_true, dtype=int)
    candidate = np.asarray(candidate_probabilities, dtype=float)
    reference = np.asarray(reference_probabilities, dtype=float)
    groups = np.asarray(groups)
    if candidate.shape != reference.shape or candidate.shape != (len(y_true), class_count):
        raise CredibilityError("paired comparison probability shapes do not match labels")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise CredibilityError("paired group inference requires at least two independent groups")
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "comparison": "candidate_minus_reference",
        "group_count": int(len(unique_groups)),
        "bootstrap_replicates": int(bootstrap_replicates),
        "permutation_replicates": int(permutation_replicates),
    }
    for metric in ("accuracy", "macro_f1"):
        observed = _metric_value(y_true, candidate, metric, class_count) - _metric_value(
            y_true, reference, metric, class_count
        )
        bootstrap_deltas: list[float] = []
        for _ in range(bootstrap_replicates):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled_indices = np.concatenate([by_group[group] for group in sampled_groups])
            bootstrap_deltas.append(
                _metric_value(y_true[sampled_indices], candidate[sampled_indices], metric, class_count)
                - _metric_value(y_true[sampled_indices], reference[sampled_indices], metric, class_count)
            )
        null_at_least_observed = 0
        for _ in range(permutation_replicates):
            swapped_candidate = candidate.copy()
            swapped_reference = reference.copy()
            for group in unique_groups:
                if rng.random() < 0.5:
                    indices = by_group[group]
                    swapped_candidate[indices] = reference[indices]
                    swapped_reference[indices] = candidate[indices]
            null_delta = _metric_value(y_true, swapped_candidate, metric, class_count) - _metric_value(
                y_true, swapped_reference, metric, class_count
            )
            null_at_least_observed += int(null_delta >= observed - 1e-15)
        interval = np.quantile(bootstrap_deltas, [0.025, 0.975])
        result[metric] = {
            "observed_delta": round(float(observed), 6),
            "paired_group_bootstrap_95_ci": [round(float(value), 6) for value in interval],
            "one_sided_group_swap_permutation_p": round(
                (null_at_least_observed + 1) / (permutation_replicates + 1), 6
            ),
        }
    return result


def nested_grouped_multimodal_evaluation(
    manifest: Mapping[str, Any],
    modality_features: Mapping[str, Any],
    *,
    feature_provenance: Mapping[str, Mapping[str, Any]],
    class_names: Sequence[str],
    feature_sample_ids: Sequence[str] | None = None,
    group_field: str = "session_id",
    statistical_group_field: str | None = None,
    required_identity_fields: Sequence[str] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 2026,
    bootstrap_replicates: int = 2000,
    permutation_replicates: int = 5000,
    alpha: float = 0.05,
    min_claim_groups: int = 10,
) -> dict[str, Any]:
    """Run leakage-resistant nested CV and test fusion against a nested baseline.

    The reference unimodal model is selected independently inside every outer
    training fold.  Outer labels therefore never choose the comparison baseline.
    """

    if group_field not in GROUP_IDENTITY_FIELDS:
        raise CredibilityError(f"group_field must be one of {GROUP_IDENTITY_FIELDS}")
    statistical_group_field = statistical_group_field or group_field
    if statistical_group_field not in GROUP_IDENTITY_FIELDS:
        raise CredibilityError(
            f"statistical_group_field must be one of {GROUP_IDENTITY_FIELDS}"
        )
    if outer_splits < 2 or inner_splits < 2:
        raise CredibilityError("outer_splits and inner_splits must both be at least two")
    if min_claim_groups < 2:
        raise CredibilityError("min_claim_groups must be at least two")
    records = manifest.get("records", [])
    expected_sample_ids = [str(record.get("sample_id", "")) for record in records]
    feature_order_verified = feature_sample_ids is not None
    if feature_sample_ids is not None:
        observed_sample_ids = [str(value) for value in feature_sample_ids]
        if observed_sample_ids != expected_sample_ids:
            raise CredibilityError(
                "feature_sample_ids do not exactly match manifest record order"
            )
    feature_sets, base_names, dimensions = _prepare_feature_sets(
        modality_features, sample_count=len(records)
    )
    if required_identity_fields is None:
        required_identity_fields = IDENTITY_FIELDS
    normalized_required_identities = tuple(
        dict.fromkeys([*required_identity_fields, group_field, statistical_group_field])
    )
    manifest_summary = validate_strict_manifest(
        manifest,
        required_modalities=base_names,
        required_identity_fields=normalized_required_identities,
    )
    normalized_feature_provenance = validate_feature_provenance(
        feature_provenance, base_names
    )
    np, _, StratifiedGroupKFold, _, _ = _dependencies()
    labels = np.asarray([int(record["label"]) for record in records], dtype=int)
    normalized_class_names = validate_class_schema(
        class_names,
        records,
        expected_class_count=manifest_summary["class_count"],
    )
    groups = np.asarray([str(record[group_field]) for record in records])
    statistical_groups = np.asarray([str(record[statistical_group_field]) for record in records])
    _assert_class_group_support(labels, groups, splits=outer_splits, scope="outer CV")

    all_names = base_names + ["fusion"]
    probabilities = {
        name: np.zeros((len(labels), len(class_names)), dtype=float) for name in all_names
    }
    best_unimodal_probabilities = np.zeros((len(labels), len(class_names)), dtype=float)
    assigned = np.zeros(len(labels), dtype=bool)
    fold_ids = np.zeros(len(labels), dtype=int)
    outer_reports: list[dict[str, Any]] = []
    splitter = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(feature_sets["fusion"], labels, groups), 1
    ):
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        overlap = sorted(train_groups & test_groups)
        if overlap:
            raise RuntimeError(f"group leakage in outer fold {fold_index}: {overlap}")
        _assert_class_group_support(
            labels[train_indices], groups[train_indices], splits=inner_splits, scope=f"outer fold {fold_index} inner CV"
        )
        modality_reports: dict[str, Any] = {}
        for modality in all_names:
            best_c, inner_score, candidates = _select_c_nested(
                feature_sets[modality][train_indices],
                labels[train_indices],
                groups[train_indices],
                class_count=len(class_names),
                inner_splits=inner_splits,
                c_grid=c_grid,
                seed=seed + fold_index,
            )
            fold_probabilities, _, _ = _fit_predict(
                feature_sets[modality][train_indices],
                labels[train_indices],
                feature_sets[modality][test_indices],
                c_value=best_c,
                class_count=len(class_names),
            )
            probabilities[modality][test_indices] = fold_probabilities
            modality_reports[modality] = {
                "selected_C": best_c,
                "inner_oof_macro_f1": round(inner_score, 6),
                "hyperparameter_candidates": candidates,
                "outer_test_metrics": classification_metrics(
                    labels[test_indices], fold_probabilities, class_names=list(normalized_class_names)
                ),
            }
        selected_unimodal = sorted(
            base_names,
            key=lambda name: (-modality_reports[name]["inner_oof_macro_f1"], name),
        )[0]
        best_unimodal_probabilities[test_indices] = probabilities[selected_unimodal][test_indices]
        assigned[test_indices] = True
        fold_ids[test_indices] = fold_index
        identity_overlap = {
            field_name: sorted(
                set(str(records[index][field_name]) for index in train_indices)
                & set(str(records[index][field_name]) for index in test_indices)
            )
            for field_name in normalized_required_identities
        }
        outer_reports.append(
            {
                "fold": fold_index,
                "train_count": int(len(train_indices)),
                "test_count": int(len(test_indices)),
                "train_group_count": len(train_groups),
                "test_group_count": len(test_groups),
                "train_groups": sorted(str(value) for value in train_groups),
                "test_groups": sorted(str(value) for value in test_groups),
                "group_overlap": overlap,
                "identity_overlap": identity_overlap,
                "teacher_overlap": identity_overlap.get("teacher_id", []),
                "site_overlap": identity_overlap.get("site_id", []),
                "participant_overlap": identity_overlap.get("participant_id", []),
                "cohort_overlap": identity_overlap.get("cohort_id", []),
                "best_unimodal_selected_inside_training_fold": selected_unimodal,
                "split_seed": seed,
                "modalities": modality_reports,
            }
        )
    if not assigned.all():
        raise RuntimeError("outer CV did not assign every example")

    metrics_by_modality = {
        name: classification_metrics(labels, values, class_names=list(normalized_class_names))
        for name, values in probabilities.items()
    }
    metrics_by_modality["best_unimodal_nested"] = classification_metrics(
        labels, best_unimodal_probabilities, class_names=list(normalized_class_names)
    )
    interval_probabilities = {
        **probabilities,
        "best_unimodal_nested": best_unimodal_probabilities,
    }
    metric_intervals_by_modality = {
        name: _grouped_metric_intervals(
            labels,
            values,
            statistical_groups,
            class_count=len(class_names),
            replicates=bootstrap_replicates,
            seed=seed + 10_000 + index,
        )
        for index, (name, values) in enumerate(interval_probabilities.items())
    }
    paired = paired_group_comparison(
        labels,
        probabilities["fusion"],
        best_unimodal_probabilities,
        statistical_groups,
        class_count=len(class_names),
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        seed=seed,
    )
    enough_groups = len(set(statistical_groups)) >= min_claim_groups
    macro_test = paired["macro_f1"]
    cross_group_valid = bool(len(set(groups)) >= min_claim_groups)
    gain_established = bool(
        enough_groups
        and cross_group_valid
        and feature_order_verified
        and macro_test["observed_delta"] > 0
        and macro_test["paired_group_bootstrap_95_ci"][0] > 0
        and macro_test["one_sided_group_swap_permutation_p"] < alpha
    )
    fusion_metrics = metrics_by_modality["fusion"]
    fusion_intervals = metric_intervals_by_modality["fusion"]
    per_class = fusion_metrics.get("per_class", {})
    minimum_recall = min(float(value["recall"]) for value in per_class.values())
    minimum_support = min(int(value["support"]) for value in per_class.values())

    def accuracy_gate(observed: Any, required: Any, passed: bool) -> dict[str, Any]:
        return {"observed": observed, "required": required, "passed": bool(passed)}

    accuracy_claim_gates = {
        "feature_sample_order": accuracy_gate(
            feature_order_verified, True, feature_order_verified
        ),
        "minimum_outer_groups": accuracy_gate(
            len(set(groups)), min_claim_groups, cross_group_valid
        ),
        "minimum_statistical_groups": accuracy_gate(
            len(set(statistical_groups)), min_claim_groups, enough_groups
        ),
        "accuracy": accuracy_gate(
            float(fusion_metrics["accuracy"]),
            GROUPED_ACCURACY_THRESHOLDS["minimum_accuracy"],
            float(fusion_metrics["accuracy"])
            >= GROUPED_ACCURACY_THRESHOLDS["minimum_accuracy"],
        ),
        "macro_f1": accuracy_gate(
            float(fusion_metrics["macro_f1"]),
            GROUPED_ACCURACY_THRESHOLDS["minimum_macro_f1"],
            float(fusion_metrics["macro_f1"])
            >= GROUPED_ACCURACY_THRESHOLDS["minimum_macro_f1"],
        ),
        "accuracy_ci_lower": accuracy_gate(
            float(fusion_intervals["accuracy_95_ci"][0]),
            GROUPED_ACCURACY_THRESHOLDS["minimum_accuracy_ci_lower"],
            float(fusion_intervals["accuracy_95_ci"][0])
            >= GROUPED_ACCURACY_THRESHOLDS["minimum_accuracy_ci_lower"],
        ),
        "macro_f1_ci_lower": accuracy_gate(
            float(fusion_intervals["macro_f1_95_ci"][0]),
            GROUPED_ACCURACY_THRESHOLDS["minimum_macro_f1_ci_lower"],
            float(fusion_intervals["macro_f1_95_ci"][0])
            >= GROUPED_ACCURACY_THRESHOLDS["minimum_macro_f1_ci_lower"],
        ),
        "minimum_per_class_recall": accuracy_gate(
            minimum_recall,
            GROUPED_ACCURACY_THRESHOLDS["minimum_per_class_recall"],
            minimum_recall
            >= GROUPED_ACCURACY_THRESHOLDS["minimum_per_class_recall"],
        ),
        "minimum_per_class_support": accuracy_gate(
            minimum_support,
            GROUPED_ACCURACY_THRESHOLDS["minimum_per_class_support"],
            minimum_support
            >= GROUPED_ACCURACY_THRESHOLDS["minimum_per_class_support"],
        ),
    }
    accuracy_established = all(
        bool(value["passed"]) for value in accuracy_claim_gates.values()
    )
    oof_rows = []
    for index, record in enumerate(records):
        oof_rows.append(
            {
                "sample_id": record["sample_id"],
                "outer_fold": int(fold_ids[index]),
                group_field: record[group_field],
                "label": int(labels[index]),
                **{
                    f"{name}_probabilities": [round(float(value), 8) for value in matrix[index]]
                    for name, matrix in probabilities.items()
                },
                "best_unimodal_nested_probabilities": [
                    round(float(value), 8) for value in best_unimodal_probabilities[index]
                ],
            }
        )
    return {
        "protocol": "nested_grouped_multimodal_evaluation_v1",
        "manifest": manifest_summary,
        "class_names": list(normalized_class_names),
        "label_semantics_verified": True,
        "base_modalities": base_names,
        "feature_dimensions": dimensions,
        "feature_provenance": normalized_feature_provenance,
        "feature_sample_order_verified": feature_order_verified,
        "outer_group_field": group_field,
        "statistical_group_field": statistical_group_field,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "seed": seed,
        "group_count": len(set(groups)),
        "statistical_group_count": len(set(statistical_groups)),
        "minimum_groups_for_claim": min_claim_groups,
        "outer_group_overlap_detected": False,
        "hyperparameters_selected_without_outer_test_labels": True,
        "best_unimodal_selected_without_outer_test_labels": True,
        "metrics": metrics_by_modality,
        "cluster_bootstrap_95_intervals": metric_intervals_by_modality,
        "fusion_vs_best_unimodal_nested": paired,
        "multimodal_gain_established": gain_established,
        "accuracy_claim_thresholds": dict(GROUPED_ACCURACY_THRESHOLDS),
        "accuracy_claim_gates": accuracy_claim_gates,
        "cross_group_evaluation_valid": cross_group_valid,
        "cross_group_accuracy_established": accuracy_established,
        "session_disjoint_accuracy_established": accuracy_established and group_field == "session_id",
        "teacher_disjoint_accuracy_established": accuracy_established and group_field == "teacher_id",
        "participant_disjoint_accuracy_established": (
            accuracy_established and group_field == "participant_id"
        ),
        "cohort_disjoint_accuracy_established": accuracy_established and group_field == "cohort_id",
        "site_disjoint_accuracy_established": accuracy_established and group_field == "site_id",
        "claim_rule": (
            "strict verified metadata and label schema; exact feature row order; >= minimum "
            "independent groups; nested group CV; absolute Accuracy/Macro-F1, cluster-CI, "
            "per-class recall/support gates; multimodal gain additionally requires fusion "
            "macro-F1 delta paired-bootstrap lower bound > 0 and one-sided group-swap p < alpha"
        ),
        "alpha": alpha,
        "folds": outer_reports,
        "oof_predictions": oof_rows,
    }


@dataclass(frozen=True)
class FrozenDeploymentModel:
    """A source-only fitted model whose schema and training population are pinned."""

    modality: str
    component_modalities: tuple[str, ...]
    feature_names_by_modality: dict[str, tuple[str, ...]]
    feature_provenance_by_modality: dict[str, dict[str, Any]]
    class_names: tuple[str, ...]
    selected_c: float
    required_identity_fields: tuple[str, ...]
    claim_cluster_field: str
    training_identity_values_by_field: dict[str, frozenset[str]]
    claim_contract: ClaimContract
    training_dataset_fingerprint: str
    training_feature_bundle_fingerprint: str
    training_feature_bundle_algorithm: str
    training_sample_ids: frozenset[str]
    training_content_hashes: frozenset[str]
    model_fingerprint: str
    selection_report: tuple[dict[str, Any], ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficient: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    model_classes: tuple[int, ...]


def _frozen_model_payload(model: FrozenDeploymentModel) -> dict[str, Any]:
    return {
        "schema_version": "3.0",
        "protocol": "frozen_deployment_model_v3_feature_and_claim_contract",
        "modality": model.modality,
        "component_modalities": list(model.component_modalities),
        "feature_names_by_modality": {
            name: list(values) for name, values in model.feature_names_by_modality.items()
        },
        "feature_provenance_by_modality": model.feature_provenance_by_modality,
        "class_names": list(model.class_names),
        "selected_C": model.selected_c,
        "required_identity_fields": list(model.required_identity_fields),
        "claim_cluster_field": model.claim_cluster_field,
        "training_identity_values_by_field": {
            name: sorted(values)
            for name, values in model.training_identity_values_by_field.items()
        },
        "claim_contract": _claim_contract_payload(model.claim_contract),
        "claim_contract_fingerprint": _claim_contract_fingerprint(
            model.claim_contract
        ),
        "training_dataset_fingerprint": model.training_dataset_fingerprint,
        "training_feature_bundle_fingerprint": model.training_feature_bundle_fingerprint,
        "training_feature_bundle_algorithm": model.training_feature_bundle_algorithm,
        "training_sample_ids": sorted(model.training_sample_ids),
        "training_content_hashes": sorted(model.training_content_hashes),
        "selection_report": list(model.selection_report),
        "scaler_mean": list(model.scaler_mean),
        "scaler_scale": list(model.scaler_scale),
        "coefficient": [list(row) for row in model.coefficient],
        "intercept": list(model.intercept),
        "model_classes": list(model.model_classes),
    }


def frozen_model_to_artifact(model: FrozenDeploymentModel) -> dict[str, Any]:
    """Return a JSON-serializable, integrity-protected frozen checkpoint."""

    payload = _frozen_model_payload(model)
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if fingerprint != model.model_fingerprint:
        raise CredibilityError("frozen model state no longer matches its fingerprint")
    return {**payload, "model_fingerprint": fingerprint}


def frozen_evaluation_report_fingerprint(report: Mapping[str, Any]) -> str:
    """Hash a completed report excluding its derived hash and receipt wrapper."""

    payload = {
        key: value
        for key, value in report.items()
        if key not in {"evaluation_report_fingerprint", "one_time_consumption_receipt"}
    }
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CredibilityError("frozen evaluation report is not canonical JSON") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_frozen_deployment_model(artifact: Mapping[str, Any]) -> FrozenDeploymentModel:
    """Load a frozen JSON checkpoint after verifying all integrity-bound fields."""

    payload = {key: value for key, value in artifact.items() if key != "model_fingerprint"}
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if artifact.get("model_fingerprint") != expected:
        raise CredibilityError("frozen model artifact fingerprint mismatch")
    if (
        payload.get("schema_version") != "3.0"
        or payload.get("protocol")
        != "frozen_deployment_model_v3_feature_and_claim_contract"
    ):
        raise CredibilityError("unsupported frozen model artifact schema")
    try:
        components = tuple(str(value) for value in payload["component_modalities"])
        required_identity_fields = tuple(
            str(value) for value in payload["required_identity_fields"]
        )
        claim_cluster_field = str(payload["claim_cluster_field"])
        if (
            not required_identity_fields
            or len(set(required_identity_fields)) != len(required_identity_fields)
            or any(name not in GROUP_IDENTITY_FIELDS for name in required_identity_fields)
            or "session_id" not in required_identity_fields
            or claim_cluster_field not in required_identity_fields
        ):
            raise CredibilityError("invalid frozen identity contract")
        raw_training_identities = payload["training_identity_values_by_field"]
        if not isinstance(raw_training_identities, Mapping):
            raise CredibilityError("malformed frozen identity populations")
        training_identity_values: dict[str, frozenset[str]] = {}
        for name, values in raw_training_identities.items():
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or not values
            ):
                raise CredibilityError("malformed frozen identity populations")
            normalized_values = frozenset(str(value).strip() for value in values)
            if not normalized_values or "" in normalized_values:
                raise CredibilityError("malformed frozen identity populations")
            training_identity_values[str(name)] = normalized_values
        if set(training_identity_values) != set(required_identity_fields) or any(
            not values for values in training_identity_values.values()
        ):
            raise CredibilityError("frozen identity populations do not match the contract")
        claim_contract = _normalize_claim_contract(payload["claim_contract"])
        if payload.get("claim_contract_fingerprint") != _claim_contract_fingerprint(
            claim_contract
        ):
            raise CredibilityError("frozen claim contract fingerprint mismatch")
        feature_names = {
            str(name): tuple(str(value) for value in values)
            for name, values in payload["feature_names_by_modality"].items()
        }
        feature_provenance = validate_feature_provenance(
            payload["feature_provenance_by_modality"], components
        )
        loaded_class_names = tuple(payload["class_names"])
        if (
            not loaded_class_names
            or any(
                not isinstance(value, str) or not value.strip()
                for value in loaded_class_names
            )
            or len(set(loaded_class_names)) != len(loaded_class_names)
        ):
            raise CredibilityError("invalid frozen class schema")
        model = FrozenDeploymentModel(
            modality=str(payload["modality"]),
            component_modalities=components,
            feature_names_by_modality=feature_names,
            feature_provenance_by_modality=feature_provenance,
            class_names=tuple(value.strip() for value in loaded_class_names),
            selected_c=float(payload["selected_C"]),
            required_identity_fields=required_identity_fields,
            claim_cluster_field=claim_cluster_field,
            training_identity_values_by_field=training_identity_values,
            claim_contract=claim_contract,
            training_dataset_fingerprint=str(payload["training_dataset_fingerprint"]),
            training_feature_bundle_fingerprint=str(
                payload["training_feature_bundle_fingerprint"]
            ),
            training_feature_bundle_algorithm=str(
                payload["training_feature_bundle_algorithm"]
            ),
            training_sample_ids=frozenset(str(value) for value in payload["training_sample_ids"]),
            training_content_hashes=frozenset(
                str(value) for value in payload["training_content_hashes"]
            ),
            model_fingerprint=expected,
            selection_report=tuple(dict(value) for value in payload["selection_report"]),
            scaler_mean=tuple(float(value) for value in payload["scaler_mean"]),
            scaler_scale=tuple(float(value) for value in payload["scaler_scale"]),
            coefficient=tuple(
                tuple(float(value) for value in row) for row in payload["coefficient"]
            ),
            intercept=tuple(float(value) for value in payload["intercept"]),
            model_classes=tuple(int(value) for value in payload["model_classes"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CredibilityError("malformed frozen model artifact") from exc
    dimensions = sum(len(model.feature_names_by_modality[name]) for name in components)
    if (
        not components
        or set(feature_names) != set(components)
        or len(model.scaler_mean) != dimensions
        or len(model.scaler_scale) != dimensions
        or any(len(row) != dimensions for row in model.coefficient)
        or len(model.intercept) != len(model.coefficient)
        or len(model.model_classes) != len(model.class_names)
        or len(model.training_dataset_fingerprint) != 64
        or len(model.training_feature_bundle_fingerprint) != 64
        or model.training_feature_bundle_algorithm != "strict_feature_bundle_v1"
    ):
        raise CredibilityError("inconsistent dimensions in frozen model artifact")
    return model


def _predict_frozen_probabilities(model: FrozenDeploymentModel, matrix: Any) -> Any:
    np, _, _, _, _ = _dependencies()
    mean = np.asarray(model.scaler_mean, dtype=float)
    scale = np.asarray(model.scaler_scale, dtype=float)
    coefficient = np.asarray(model.coefficient, dtype=float)
    intercept = np.asarray(model.intercept, dtype=float)
    scaled = (matrix - mean) / scale
    scores = scaled @ coefficient.T + intercept
    if coefficient.shape[0] == 1 and len(model.model_classes) == 2:
        positive = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        raw = np.column_stack([1.0 - positive, positive])
    else:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        raw = exponent / exponent.sum(axis=1, keepdims=True)
    aligned = np.zeros((len(matrix), len(model.class_names)), dtype=float)
    for column, label in enumerate(model.model_classes):
        aligned[:, label] = raw[:, column]
    return aligned


def _matrix_for_frozen_model(
    model: FrozenDeploymentModel,
    modality_features: Mapping[str, Any],
    feature_names_by_modality: Mapping[str, Sequence[str]],
    feature_provenance: Mapping[str, Mapping[str, Any]],
    *,
    sample_count: int,
) -> Any:
    np, _, _, _, _ = _dependencies()
    observed_provenance = validate_feature_provenance(
        feature_provenance, model.component_modalities
    )
    if observed_provenance != model.feature_provenance_by_modality:
        raise CredibilityError("feature extractor provenance does not match the frozen model")
    matrices = []
    for name in model.component_modalities:
        observed_names = tuple(str(value) for value in feature_names_by_modality.get(name, ()))
        if observed_names != model.feature_names_by_modality[name]:
            raise CredibilityError(f"feature schema mismatch for modality {name}")
        if name not in modality_features:
            raise CredibilityError(f"missing frozen-evaluation features for modality {name}")
        matrix = np.asarray(modality_features[name], dtype=float)
        expected_dimension = len(observed_names)
        if matrix.shape != (sample_count, expected_dimension) or not np.isfinite(matrix).all():
            raise CredibilityError(f"invalid frozen-evaluation feature matrix for modality {name}")
        matrices.append(matrix)
    return matrices[0] if len(matrices) == 1 else np.concatenate(matrices, axis=1)


def fit_frozen_deployment_model(
    source_manifest: Mapping[str, Any],
    modality_features: Mapping[str, Any],
    feature_names_by_modality: Mapping[str, Sequence[str]],
    *,
    feature_provenance: Mapping[str, Mapping[str, Any]],
    class_names: Sequence[str],
    modality: str = "fusion",
    group_field: str = "session_id",
    claim_cluster_field: str = "session_id",
    required_identity_fields: Sequence[str] = IDENTITY_FIELDS,
    claim_contract: ClaimContract | Mapping[str, Any] | None = None,
    inner_splits: int = 5,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 2026,
) -> FrozenDeploymentModel:
    """Select and fit a model using source data only, before external labels are seen."""

    identity_contract = tuple(
        dict.fromkeys(str(name) for name in required_identity_fields)
    )
    invalid_identity_fields = [
        name for name in identity_contract if name not in GROUP_IDENTITY_FIELDS
    ]
    if not identity_contract or invalid_identity_fields:
        raise CredibilityError(
            "invalid required_identity_fields: " + ", ".join(invalid_identity_fields)
        )
    if "session_id" not in identity_contract:
        raise CredibilityError(
            "frozen external evaluation requires session_id in the identity contract"
        )
    if group_field not in identity_contract:
        raise CredibilityError("group_field must be included in required_identity_fields")
    if claim_cluster_field not in identity_contract:
        raise CredibilityError(
            "claim_cluster_field must be included in required_identity_fields"
        )
    frozen_claim_contract = _normalize_claim_contract(claim_contract)
    records = source_manifest.get("records", [])
    feature_sets, base_names, _ = _prepare_feature_sets(
        modality_features, sample_count=len(records)
    )
    manifest_summary = validate_strict_manifest(
        source_manifest,
        required_modalities=base_names,
        required_identity_fields=identity_contract,
    )
    if modality != "fusion" and modality not in base_names:
        raise CredibilityError(f"unknown modality: {modality}")
    all_feature_provenance = validate_feature_provenance(
        feature_provenance, base_names
    )
    training_feature_bundle_fingerprint = strict_feature_bundle_fingerprint(
        records,
        {name: feature_names_by_modality.get(name, ()) for name in base_names},
        {name: modality_features[name] for name in base_names},
        all_feature_provenance,
    )
    components = tuple(base_names if modality == "fusion" else [modality])
    normalized_feature_provenance = {
        name: all_feature_provenance[name] for name in components
    }
    normalized_feature_names: dict[str, tuple[str, ...]] = {}
    for name in components:
        names = tuple(str(value) for value in feature_names_by_modality.get(name, ()))
        if len(names) != feature_sets[name].shape[1] or len(set(names)) != len(names):
            raise CredibilityError(f"feature_names_by_modality does not match {name} matrix")
        normalized_feature_names[name] = names
    np, _, _, _, _ = _dependencies()
    labels = np.asarray([int(record["label"]) for record in records], dtype=int)
    normalized_class_names = validate_class_schema(
        class_names,
        records,
        expected_class_count=manifest_summary["class_count"],
    )
    groups = np.asarray([str(record[group_field]) for record in records])
    best_c, _, candidates = _select_c_nested(
        feature_sets[modality],
        labels,
        groups,
        class_count=len(class_names),
        inner_splits=inner_splits,
        c_grid=c_grid,
        seed=seed,
    )
    _, scaler, classifier = _fit_predict(
        feature_sets[modality],
        labels,
        feature_sets[modality][:1],
        c_value=best_c,
        class_count=len(class_names),
    )
    provisional = FrozenDeploymentModel(
        modality=modality,
        component_modalities=components,
        feature_names_by_modality=normalized_feature_names,
        feature_provenance_by_modality=normalized_feature_provenance,
        class_names=normalized_class_names,
        selected_c=best_c,
        required_identity_fields=identity_contract,
        claim_cluster_field=claim_cluster_field,
        training_identity_values_by_field={
            name: frozenset(str(record[name]) for record in records)
            for name in identity_contract
        },
        claim_contract=frozen_claim_contract,
        training_dataset_fingerprint=manifest_summary["dataset_fingerprint"],
        training_feature_bundle_fingerprint=training_feature_bundle_fingerprint,
        training_feature_bundle_algorithm="strict_feature_bundle_v1",
        training_sample_ids=frozenset(str(record["sample_id"]) for record in records),
        training_content_hashes=frozenset(str(record["content_sha256"]) for record in records),
        model_fingerprint="",
        selection_report=tuple(candidates),
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        coefficient=tuple(tuple(float(value) for value in row) for row in classifier.coef_),
        intercept=tuple(float(value) for value in classifier.intercept_),
        model_classes=tuple(int(value) for value in classifier.classes_),
    )
    model_fingerprint = hashlib.sha256(
        json.dumps(
            _frozen_model_payload(provisional),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return replace(provisional, model_fingerprint=model_fingerprint)


def _grouped_metric_intervals(
    labels: Any,
    probabilities: Any,
    groups: Any,
    *,
    class_count: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    np, _, _, _, _ = _dependencies()
    if replicates < 1:
        raise CredibilityError("bootstrap replicate count must be positive")
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    draws = {"accuracy": [], "macro_f1": []}
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        for metric in draws:
            draws[metric].append(
                _metric_value(labels[indices], probabilities[indices], metric, class_count)
            )
    return {
        f"{metric}_95_ci": [
            round(float(value), 6) for value in np.quantile(values, [0.025, 0.975])
        ]
        for metric, values in draws.items()
    }


def _claim_contract_fingerprint(contract: ClaimContract) -> str:
    canonical = json.dumps(
        _claim_contract_payload(contract),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _claim_gate_report(
    contract: ClaimContract,
    *,
    metrics: Mapping[str, Any],
    intervals: Mapping[str, Sequence[float]],
    session_count: int,
    coverage_fraction: float | None,
    prospective: bool,
    one_time_lockbox: bool,
    independent_cluster_structure: bool,
) -> dict[str, Any]:
    primary_value = float(metrics[contract.primary_metric])
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping) or not per_class:
        raise CredibilityError("external metrics lack per-class results")
    recalls = [float(value["recall"]) for value in per_class.values()]
    supports = [int(value["support"]) for value in per_class.values()]
    accuracy_ci_lower = float(intervals["accuracy_95_ci"][0])
    macro_f1_ci_lower = float(intervals["macro_f1_95_ci"][0])

    def gate(observed: Any, required: Any, passed: bool) -> dict[str, Any]:
        return {"observed": observed, "required": required, "passed": bool(passed)}

    gates = {
        "primary_metric": {
            "metric": contract.primary_metric,
            **gate(
                primary_value,
                contract.minimum_primary_metric,
                primary_value >= contract.minimum_primary_metric,
            ),
        },
        "accuracy": gate(
            float(metrics["accuracy"]),
            contract.minimum_accuracy,
            float(metrics["accuracy"]) >= contract.minimum_accuracy,
        ),
        "macro_f1": gate(
            float(metrics["macro_f1"]),
            contract.minimum_macro_f1,
            float(metrics["macro_f1"]) >= contract.minimum_macro_f1,
        ),
        "accuracy_ci_lower": gate(
            accuracy_ci_lower,
            contract.minimum_accuracy_ci_lower,
            accuracy_ci_lower >= contract.minimum_accuracy_ci_lower,
        ),
        "macro_f1_ci_lower": gate(
            macro_f1_ci_lower,
            contract.minimum_macro_f1_ci_lower,
            macro_f1_ci_lower >= contract.minimum_macro_f1_ci_lower,
        ),
        "minimum_per_class_recall": gate(
            min(recalls),
            contract.minimum_per_class_recall,
            min(recalls) >= contract.minimum_per_class_recall,
        ),
        "minimum_per_class_support": gate(
            min(supports),
            contract.minimum_per_class_support,
            min(supports) >= contract.minimum_per_class_support,
        ),
        "minimum_sessions": gate(
            int(session_count),
            contract.minimum_sessions,
            int(session_count) >= contract.minimum_sessions,
        ),
        "independent_cluster_structure": gate(
            independent_cluster_structure,
            True,
            independent_cluster_structure,
        ),
        "minimum_coverage_fraction": gate(
            coverage_fraction,
            contract.minimum_coverage_fraction,
            coverage_fraction is not None
            and coverage_fraction >= contract.minimum_coverage_fraction,
        ),
        "prospective_collection": gate(
            prospective,
            contract.require_prospective,
            prospective or not contract.require_prospective,
        ),
        "one_time_lockbox": gate(
            one_time_lockbox,
            contract.require_one_time_lockbox,
            one_time_lockbox or not contract.require_one_time_lockbox,
        ),
    }
    return {
        "gates": gates,
        "all_gates_passed": all(bool(value["passed"]) for value in gates.values()),
    }


def evaluate_frozen_external_deployment(
    model: FrozenDeploymentModel,
    external_manifest: Mapping[str, Any],
    modality_features: Mapping[str, Any],
    feature_names_by_modality: Mapping[str, Sequence[str]],
    *,
    feature_provenance: Mapping[str, Mapping[str, Any]],
    bootstrap_replicates: int = 2000,
    seed: int = 2026,
    min_claim_sessions: int | None = None,
    registration_attestation: Mapping[str, Any] | None = None,
    trusted_public_key_pem: bytes | None = None,
    one_time_ledger_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate once against the identity and claim contracts frozen in ``model``."""

    # Re-hash the in-memory checkpoint before accepting any external result.
    frozen_model_to_artifact(model)
    contract = model.claim_contract
    if min_claim_sessions is not None and min_claim_sessions != contract.minimum_sessions:
        raise CredibilityError(
            "min_claim_sessions cannot override the frozen claim contract"
        )
    records = external_manifest.get("records", [])
    summary = validate_strict_manifest(
        external_manifest,
        required_modalities=model.component_modalities,
        required_identity_fields=model.required_identity_fields,
    )
    if summary["class_count"] != len(model.class_names):
        raise CredibilityError("external labels do not match the frozen model class schema")
    validate_class_schema(
        model.class_names,
        records,
        expected_class_count=summary["class_count"],
    )
    audit = external_manifest["audit"]
    if audit.get("held_out_from_model_development") is not True:
        raise CredibilityError("external dataset is not documented as held out from model development")
    if audit.get("deployment_target_documented") is not True:
        raise CredibilityError("external dataset lacks a documented deployment target population")
    if summary["dataset_fingerprint"] == model.training_dataset_fingerprint:
        raise CredibilityError("external and training dataset fingerprints are identical")
    external_feature_bundle_fingerprint = strict_feature_bundle_fingerprint(
        records,
        feature_names_by_modality,
        modality_features,
        feature_provenance,
    )
    coverage_summary = validate_evaluation_coverage(
        audit,
        evaluated_sample_count=len(records),
    )
    external_sets: dict[str, set[str]] = {
        "sample": {str(record["sample_id"]) for record in records},
        "content": {str(record["content_sha256"]) for record in records},
        **{
            name: {str(record[name]) for record in records}
            for name in model.required_identity_fields
        },
    }
    training_sets: dict[str, frozenset[str]] = {
        "sample": model.training_sample_ids,
        "content": model.training_content_hashes,
        **model.training_identity_values_by_field,
    }
    overlaps = {
        name: sorted(training_sets[name] & external_sets[name]) for name in external_sets
    }
    nonempty = {name: values for name, values in overlaps.items() if values}
    if nonempty:
        raise CredibilityError(f"external holdout overlaps training identities/content: {nonempty}")

    registration_verification: dict[str, Any] = {
        "verified": False,
        "reason": "no independently signed freeze registration was supplied",
    }
    consumption_receipt: dict[str, Any] | None = None
    registration_inputs = (
        registration_attestation,
        trusted_public_key_pem,
        one_time_ledger_dir,
    )
    if any(value is not None for value in registration_inputs):
        if any(value is None for value in registration_inputs):
            raise CredibilityError(
                "registration_attestation, trusted_public_key_pem, and one_time_ledger_dir "
                "must be supplied together"
            )
        registration_verification = verify_freeze_registration(
            registration_attestation or {},
            trusted_public_key_pem=trusted_public_key_pem or b"",
            expected_bindings={
                "model_fingerprint": model.model_fingerprint,
                "claim_contract_fingerprint": _claim_contract_fingerprint(contract),
                "training_dataset_fingerprint": model.training_dataset_fingerprint,
                "training_feature_bundle_fingerprint": (
                    model.training_feature_bundle_fingerprint
                ),
                "external_dataset_fingerprint": summary["dataset_fingerprint"],
                "external_feature_bundle_fingerprint": (
                    external_feature_bundle_fingerprint
                ),
                "external_coverage_evidence_fingerprint": coverage_summary[
                    "coverage_evidence_fingerprint"
                ],
            },
        )
        consumption_receipt = consume_registration_once(
            one_time_ledger_dir or "",
            registration_verification,
        )

    matrix = _matrix_for_frozen_model(
        model,
        modality_features,
        feature_names_by_modality,
        feature_provenance,
        sample_count=len(records),
    )
    probabilities = _predict_frozen_probabilities(model, matrix)
    np, _, _, _, _ = _dependencies()
    labels = np.asarray([int(record["label"]) for record in records], dtype=int)
    claim_cluster_groups = np.asarray(
        [str(record[model.claim_cluster_field]) for record in records]
    )
    cross_cluster_person_dependencies: dict[str, dict[str, list[str]]] = {}
    for identity_field in ("participant_id", "teacher_id"):
        if (
            identity_field not in model.required_identity_fields
            or identity_field == model.claim_cluster_field
        ):
            continue
        clusters_by_identity: dict[str, set[str]] = {}
        for record in records:
            clusters_by_identity.setdefault(str(record[identity_field]), set()).add(
                str(record[model.claim_cluster_field])
            )
        repeated = {
            identity: sorted(clusters)
            for identity, clusters in clusters_by_identity.items()
            if len(clusters) > 1
        }
        if repeated:
            cross_cluster_person_dependencies[identity_field] = repeated
    independent_cluster_structure = not cross_cluster_person_dependencies
    metrics = classification_metrics(labels, probabilities, class_names=list(model.class_names))
    intervals = _grouped_metric_intervals(
        labels,
        probabilities,
        claim_cluster_groups,
        class_count=len(model.class_names),
        replicates=bootstrap_replicates,
        seed=seed,
    )
    manifest_prospective = audit.get("prospective_deployment_collection") is True
    manifest_one_time = audit.get("one_time_lockbox_evaluation") is True
    prospective = bool(
        manifest_prospective
        and registration_verification.get("verified") is True
        and registration_verification.get("custodian_attestations", {}).get(
            "prospective_collection"
        )
        is True
    )
    one_time_lockbox = bool(
        manifest_one_time
        and registration_verification.get("verified") is True
        and consumption_receipt is not None
    )
    coverage_fraction = float(coverage_summary["coverage_fraction"])
    claim = _claim_gate_report(
        contract,
        metrics=metrics,
        intervals=intervals,
        session_count=len(set(claim_cluster_groups)),
        coverage_fraction=coverage_fraction,
        prospective=prospective,
        one_time_lockbox=one_time_lockbox,
        independent_cluster_structure=independent_cluster_structure,
    )
    external_accuracy_established = bool(claim["all_gates_passed"])
    deployment_accuracy_established = bool(
        external_accuracy_established
        and prospective
        and one_time_lockbox
        and registration_verification.get("verified") is True
    )
    evaluation_fingerprint = hashlib.sha256(
        (
            f"{model.model_fingerprint}:{summary['dataset_fingerprint']}:"
            f"{external_feature_bundle_fingerprint}:"
            f"{_claim_contract_fingerprint(contract)}:"
            f"{registration_verification.get('attestation_fingerprint', 'unregistered')}"
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "protocol": "frozen_external_deployment_evaluation_v3_signed_registration",
        "model_fingerprint": model.model_fingerprint,
        "required_identity_fields": list(model.required_identity_fields),
        "claim_contract": _claim_contract_payload(contract),
        "claim_contract_fingerprint": _claim_contract_fingerprint(contract),
        "external_dataset_fingerprint": summary["dataset_fingerprint"],
        "external_feature_bundle_fingerprint": external_feature_bundle_fingerprint,
        "external_feature_bundle_algorithm": "strict_feature_bundle_v1",
        "training_feature_bundle_fingerprint": model.training_feature_bundle_fingerprint,
        "training_feature_bundle_algorithm": model.training_feature_bundle_algorithm,
        "evaluation_fingerprint": evaluation_fingerprint,
        "modality": model.modality,
        "training_external_overlap": overlaps,
        "external_sample_count": summary["sample_count"],
        "external_session_count": summary["session_count"],
        "claim_cluster_field": model.claim_cluster_field,
        "independent_claim_cluster_count": len(set(claim_cluster_groups)),
        "cross_cluster_person_dependencies": cross_cluster_person_dependencies,
        "independent_cluster_structure_verified": independent_cluster_structure,
        "external_identity_counts": {
            name: summary[f"{name.removesuffix('_id')}_count"]
            for name in model.required_identity_fields
        },
        "external_teacher_count": summary.get("teacher_count"),
        "external_site_count": summary.get("site_count"),
        "minimum_sessions_for_claim": contract.minimum_sessions,
        "evaluation_coverage_fraction": coverage_fraction,
        "evaluation_coverage": coverage_summary,
        "model_or_hyperparameter_fit_on_external_data": False,
        "feature_schema_exact_match": True,
        "metrics": metrics,
        "claim_cluster_bootstrap": {
            **intervals,
            "replicates": bootstrap_replicates,
            "cluster_unit": f"verified_{model.claim_cluster_field}",
        },
        "session_cluster_bootstrap": (
            {
                **intervals,
                "replicates": bootstrap_replicates,
                "cluster_unit": "verified_session_id",
            }
            if model.claim_cluster_field == "session_id"
            else None
        ),
        "claim_gate_results": claim["gates"],
        "all_frozen_claim_gates_passed": claim["all_gates_passed"],
        "external_site_frozen_accuracy_established": external_accuracy_established,
        "manifest_prospective_deployment_collection": manifest_prospective,
        "manifest_one_time_lockbox_evaluation": manifest_one_time,
        "prospective_deployment_collection": prospective,
        "one_time_lockbox_evaluation": one_time_lockbox,
        "freeze_registration": {
            key: value
            for key, value in registration_verification.items()
            if key != "custodian_attestations"
        },
        "deployment_accuracy_established": deployment_accuracy_established,
        "claim_scope": (
            "prospective target-population deployment accuracy"
            if deployment_accuracy_established
            else "external frozen-model estimate; one or more frozen claim gates failed"
        ),
        "predictions": [
            {
                "sample_id": record["sample_id"],
                "session_id": record["session_id"],
                "label": int(labels[index]),
                "probabilities": [round(float(value), 8) for value in probabilities[index]],
            }
            for index, record in enumerate(records)
        ],
    }
    if consumption_receipt is not None:
        report["evaluation_report_fingerprint"] = frozen_evaluation_report_fingerprint(
            report
        )
        finalized = finalize_consumption_receipt(
            consumption_receipt,
            evaluation_fingerprint=report["evaluation_report_fingerprint"],
        )
        report["one_time_consumption_receipt"] = {
            key: value
            for key, value in finalized.items()
            if key != "ledger_path"
        }
    else:
        report["one_time_consumption_receipt"] = None
        report["evaluation_report_fingerprint"] = frozen_evaluation_report_fingerprint(
            report
        )
    return report
