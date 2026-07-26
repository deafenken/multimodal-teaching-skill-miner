"""Fail-closed preregistration for a confirmatory TeachObs-style lockbox.

The public TeachObs 23/7 split is useful development evidence, but its test
labels were accessible before this project froze a system.  This module
therefore treats TeachObs only as the development source and describes a
separate, prospective new-site evaluation.  It freezes four model arms and a
paired classroom/session-cluster analysis without accepting result fields in
the preregistration itself.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


PROTOCOL = "teachobs_new_site_confirmatory_multimodal_preregistration_v2"
ATTESTATION_PROTOCOL = (
    "ed25519_teachobs_new_site_confirmatory_preregistration_attestation_v2"
)
ARM_ORDER = (
    "transcript_only",
    "transcript_audio",
    "transcript_visual",
    "full",
)
ARM_DEFINITIONS: dict[str, dict[str, list[str]]] = {
    "transcript_only": {
        "modalities": ["transcript"],
        "feature_blocks": ["training_only_transcript_char_tfidf"],
    },
    "transcript_audio": {
        "modalities": ["transcript", "audio"],
        "feature_blocks": [
            "training_only_transcript_char_tfidf",
            "scene_aligned_audio_numeric",
        ],
    },
    "transcript_visual": {
        "modalities": ["transcript", "visual"],
        "feature_blocks": [
            "training_only_transcript_char_tfidf",
            "training_only_ocr_char_tfidf",
            "training_only_ocr_word_tfidf",
            "scene_aligned_visual_numeric",
            "hash_bound_clip_embedding",
        ],
    },
    "full": {
        "modalities": ["transcript", "audio", "visual"],
        "feature_blocks": [
            "training_only_transcript_char_tfidf",
            "scene_aligned_audio_numeric",
            "training_only_ocr_char_tfidf",
            "training_only_ocr_word_tfidf",
            "scene_aligned_visual_numeric",
            "hash_bound_clip_embedding",
        ],
    },
}
PRIMARY_COMPARISON = {
    "name": "full_minus_transcript_only",
    "numerator_arm": "full",
    "denominator_arm": "transcript_only",
}
SECONDARY_COMPARISONS = [
    {
        "name": "transcript_audio_minus_transcript_only",
        "numerator_arm": "transcript_audio",
        "denominator_arm": "transcript_only",
    },
    {
        "name": "transcript_visual_minus_transcript_only",
        "numerator_arm": "transcript_visual",
        "denominator_arm": "transcript_only",
    },
    {
        "name": "full_minus_transcript_audio",
        "numerator_arm": "full",
        "denominator_arm": "transcript_audio",
    },
    {
        "name": "full_minus_transcript_visual",
        "numerator_arm": "full",
        "denominator_arm": "transcript_visual",
    },
]

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_SHA256 = "0" * 64
_FROZEN_BINDING_SCHEMA = (
    "teaching_skill_miner.teachobs_frozen_lockbox_artifact_set.v2"
)
_VERIFIED_FROZEN_BINDING_SCHEMA = (
    "teaching_skill_miner.teachobs_frozen_lockbox_artifact_set.v1"
)
_DECISION_THRESHOLD_POLICY = (
    "per_arm_per_label_thresholds_from_verified_frozen_numeric_state_v1"
)
_LABEL_COUNT = 39


class TeachObsLockboxError(ValueError):
    """Raised when a confirmatory lockbox preregistration is not trustworthy."""


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
        raise TeachObsLockboxError(
            "lockbox preregistration must be finite canonical JSON"
        ) from exc
    return payload.encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(name: str, value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise TeachObsLockboxError(f"{name} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise TeachObsLockboxError(f"{name} is invalid") from exc
    if parsed > datetime.now(timezone.utc):
        raise TeachObsLockboxError(f"{name} is in the future")
    return parsed


def _require_exact_keys(name: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TeachObsLockboxError(f"{name} must be one JSON object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise TeachObsLockboxError(
            f"{name} fields do not match protocol; missing={missing}, extra={extra}"
        )
    return dict(value)


def _require_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise TeachObsLockboxError(
            f"{name} must contain 6-128 safe identifier characters"
        )
    return value


def _require_sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not _SHA256_RE.fullmatch(value)
        or value == _PLACEHOLDER_SHA256
    ):
        raise TeachObsLockboxError(
            f"{name} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TeachObsLockboxError(f"{name} must be boolean")
    return value


def _require_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeachObsLockboxError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise TeachObsLockboxError(f"{name} must be finite and between zero and one")
    return numeric


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_binding(
    logical_name: str,
    path: str | Path | None,
) -> dict[str, Any]:
    if path is None:
        return {
            "logical_name": logical_name,
            "sha256": None,
            "byte_count": None,
            "bound": False,
        }
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise TeachObsLockboxError(f"artifact does not exist: {logical_name}")
    return {
        "logical_name": logical_name,
        "sha256": _sha256_file(artifact),
        "byte_count": artifact.stat().st_size,
        "bound": True,
    }


def _binding_is_complete(name: str, raw: Any) -> bool:
    value = _require_exact_keys(
        name,
        raw,
        {"logical_name", "sha256", "byte_count", "bound"},
    )
    if not isinstance(value["logical_name"], str) or not value[
        "logical_name"
    ].strip():
        raise TeachObsLockboxError(f"{name}.logical_name must be non-empty")
    bound = _require_bool(f"{name}.bound", value["bound"])
    if bound:
        _require_sha256(f"{name}.sha256", value["sha256"])
        byte_count = value["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise TeachObsLockboxError(f"{name}.byte_count must be a positive integer")
    elif value["sha256"] is not None or value["byte_count"] is not None:
        raise TeachObsLockboxError(
            f"{name} cannot carry a digest or byte count while bound=false"
        )
    return bound


def _pending_frozen_model_binding() -> dict[str, Any]:
    return {
        "schema": _FROZEN_BINDING_SCHEMA,
        "bound": False,
        "bundle_manifest_file_sha256": None,
        "bundle_fingerprint": None,
        "label_order_sha256": None,
        "training_provenance_sha256": None,
        "software_provenance_sha256": None,
        "arms": {
            arm: {
                "manifest_file_sha256": None,
                "manifest_sha256": None,
                "arrays_file_sha256": None,
                "model_thresholds_sha256": None,
                "model_threshold_count": None,
            }
            for arm in ARM_ORDER
        },
        "companion_arrays_loaded_with_allow_pickle_false": False,
        "paths_included": False,
        "artifact_set_fingerprint": None,
    }


def _frozen_binding_fingerprint(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_set_fingerprint", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _threshold_spec_from_verified_manifest(
    arm: str,
    manifest_path: str | Path,
    *,
    expected_manifest_file_sha256: str,
) -> dict[str, Any]:
    """Read the threshold spec from the exact manifest verified with its NPZ.

    ``verify_teachobs_frozen_artifact_set`` has already loaded every companion
    array with ``allow_pickle=False`` and checked its spec.  Reading the exact
    manifest bytes again here lets the lockbox bind the verified
    ``model_thresholds`` array without changing the frozen-model implementation
    or exposing either values or paths.
    """

    try:
        raw = Path(manifest_path).read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeachObsLockboxError(
            f"cannot read verified frozen arm manifest: {arm}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != expected_manifest_file_sha256:
        raise TeachObsLockboxError(
            f"verified frozen arm manifest changed during threshold binding: {arm}"
        )
    arrays = manifest.get("arrays") if isinstance(manifest, Mapping) else None
    spec = arrays.get("model_thresholds") if isinstance(arrays, Mapping) else None
    if (
        not isinstance(spec, Mapping)
        or set(spec) != {"dtype", "shape", "sha256"}
        or spec.get("dtype") != "float64"
        or spec.get("shape") != [_LABEL_COUNT]
    ):
        raise TeachObsLockboxError(
            f"verified frozen arm lacks one {_LABEL_COUNT}-label threshold array: {arm}"
        )
    return {
        "model_thresholds_sha256": _require_sha256(
            f"{arm}.arrays.model_thresholds.sha256", spec.get("sha256")
        ),
        "model_threshold_count": _LABEL_COUNT,
    }


def _verified_frozen_model_binding(
    bundle_manifest_path: str | Path,
    arm_manifest_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Upgrade the fully verified frozen-model binding to lockbox schema v2."""

    from .teachobs_frozen_model import (
        TeachObsFrozenModelError,
        verify_teachobs_frozen_artifact_set,
    )

    try:
        verified = verify_teachobs_frozen_artifact_set(
            bundle_manifest_path, arm_manifest_paths
        )
    except TeachObsFrozenModelError as exc:
        raise TeachObsLockboxError(
            f"invalid transitive frozen-model artifact set: {exc}"
        ) from exc
    if (
        not isinstance(verified, Mapping)
        or verified.get("schema") != _VERIFIED_FROZEN_BINDING_SCHEMA
        or verified.get("bound") is not True
    ):
        raise TeachObsLockboxError(
            "frozen-model verifier returned an unsupported artifact binding"
        )
    upgraded = dict(verified)
    upgraded["schema"] = _FROZEN_BINDING_SCHEMA
    raw_arms = verified.get("arms")
    if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(ARM_ORDER):
        raise TeachObsLockboxError("frozen-model verifier omitted an arm binding")
    upgraded_arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        raw_arm = raw_arms[arm]
        if not isinstance(raw_arm, Mapping):
            raise TeachObsLockboxError(f"invalid verified arm binding: {arm}")
        expected_manifest_sha256 = _require_sha256(
            f"verified.{arm}.manifest_file_sha256",
            raw_arm.get("manifest_file_sha256"),
        )
        upgraded_arms[arm] = {
            **dict(raw_arm),
            **_threshold_spec_from_verified_manifest(
                arm,
                arm_manifest_paths[arm],
                expected_manifest_file_sha256=expected_manifest_sha256,
            ),
        }
    upgraded["arms"] = upgraded_arms
    upgraded.pop("artifact_set_fingerprint", None)
    upgraded["artifact_set_fingerprint"] = _frozen_binding_fingerprint(upgraded)
    return upgraded


def _decision_threshold_policy(
    transitive_model_binding: Mapping[str, Any],
) -> dict[str, Any]:
    bound = transitive_model_binding.get("bound") is True
    arms = transitive_model_binding.get("arms")
    threshold_hashes = {
        arm: (
            arms[arm]["model_thresholds_sha256"]
            if bound and isinstance(arms, Mapping)
            else None
        )
        for arm in ARM_ORDER
    }
    return {
        "policy": _DECISION_THRESHOLD_POLICY,
        "label_count": _LABEL_COUNT,
        "one_threshold_per_label_per_arm": True,
        "threshold_values_included": False,
        "arm_model_thresholds_sha256": threshold_hashes,
        "frozen_artifact_set_fingerprint": (
            transitive_model_binding.get("artifact_set_fingerprint")
            if bound
            else None
        ),
        "binding_complete": bound,
    }


def _validate_decision_threshold_policy(
    raw: Any,
    *,
    transitive_model_binding: Mapping[str, Any],
    transitive_model_bound: bool,
) -> None:
    value = _require_exact_keys(
        "analysis_plan.decision_threshold_policy",
        raw,
        {
            "policy",
            "label_count",
            "one_threshold_per_label_per_arm",
            "threshold_values_included",
            "arm_model_thresholds_sha256",
            "frozen_artifact_set_fingerprint",
            "binding_complete",
        },
    )
    if (
        value["policy"] != _DECISION_THRESHOLD_POLICY
        or value["label_count"] != _LABEL_COUNT
        or value["one_threshold_per_label_per_arm"] is not True
        or value["threshold_values_included"] is not False
        or value["binding_complete"] is not transitive_model_bound
    ):
        raise TeachObsLockboxError(
            "analysis decision-threshold policy differs from the frozen protocol"
        )
    hashes = _require_exact_keys(
        "analysis_plan.decision_threshold_policy.arm_model_thresholds_sha256",
        value["arm_model_thresholds_sha256"],
        set(ARM_ORDER),
    )
    if transitive_model_bound:
        transitive_arms = transitive_model_binding["arms"]
        for arm in ARM_ORDER:
            _require_sha256(
                "analysis_plan.decision_threshold_policy."
                f"arm_model_thresholds_sha256.{arm}",
                hashes[arm],
            )
            if (
                hashes[arm]
                != transitive_arms[arm]["model_thresholds_sha256"]
            ):
                raise TeachObsLockboxError(
                    f"analysis threshold binding differs from frozen arm: {arm}"
                )
        _require_sha256(
            "analysis_plan.decision_threshold_policy."
            "frozen_artifact_set_fingerprint",
            value["frozen_artifact_set_fingerprint"],
        )
        if (
            value["frozen_artifact_set_fingerprint"]
            != transitive_model_binding["artifact_set_fingerprint"]
        ):
            raise TeachObsLockboxError(
                "analysis threshold policy binds a different frozen artifact set"
            )
    elif (
        any(hashes[arm] is not None for arm in ARM_ORDER)
        or value["frozen_artifact_set_fingerprint"] is not None
    ):
        raise TeachObsLockboxError(
            "pending decision-threshold policy cannot carry unverified digests"
        )


def _validate_frozen_model_binding(raw: Any) -> bool:
    value = _require_exact_keys(
        "frozen_artifacts.transitive_model_binding",
        raw,
        {
            "schema",
            "bound",
            "bundle_manifest_file_sha256",
            "bundle_fingerprint",
            "label_order_sha256",
            "training_provenance_sha256",
            "software_provenance_sha256",
            "arms",
            "companion_arrays_loaded_with_allow_pickle_false",
            "paths_included",
            "artifact_set_fingerprint",
        },
    )
    if value["schema"] != _FROZEN_BINDING_SCHEMA:
        raise TeachObsLockboxError("unsupported transitive frozen-model binding")
    bound = _require_bool(
        "frozen_artifacts.transitive_model_binding.bound", value["bound"]
    )
    arms = _require_exact_keys(
        "frozen_artifacts.transitive_model_binding.arms",
        value["arms"],
        set(ARM_ORDER),
    )
    checked_arm_values: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        checked_arm_values.append(
            _require_exact_keys(
                f"frozen_artifacts.transitive_model_binding.arms.{arm}",
                arms[arm],
                {
                    "manifest_file_sha256",
                    "manifest_sha256",
                    "arrays_file_sha256",
                    "model_thresholds_sha256",
                    "model_threshold_count",
                },
            )
        )
    if value["paths_included"] is not False:
        raise TeachObsLockboxError("transitive frozen-model binding cannot expose paths")
    digest_fields = (
        "bundle_manifest_file_sha256",
        "bundle_fingerprint",
        "label_order_sha256",
        "training_provenance_sha256",
        "software_provenance_sha256",
    )
    if bound:
        for field in digest_fields:
            _require_sha256(
                f"frozen_artifacts.transitive_model_binding.{field}", value[field]
            )
        for arm, arm_value in zip(ARM_ORDER, checked_arm_values, strict=True):
            for field in (
                "manifest_file_sha256",
                "manifest_sha256",
                "arrays_file_sha256",
                "model_thresholds_sha256",
            ):
                _require_sha256(
                    "frozen_artifacts.transitive_model_binding."
                    f"arms.{arm}.{field}",
                    arm_value[field],
                )
            if arm_value["model_threshold_count"] != _LABEL_COUNT:
                raise TeachObsLockboxError(
                    "frozen_artifacts.transitive_model_binding."
                    f"arms.{arm}.model_threshold_count must equal {_LABEL_COUNT}"
                )
        if value["companion_arrays_loaded_with_allow_pickle_false"] is not True:
            raise TeachObsLockboxError(
                "complete frozen binding must validate every companion arrays.npz"
            )
        _require_sha256(
            "frozen_artifacts.transitive_model_binding.artifact_set_fingerprint",
            value["artifact_set_fingerprint"],
        )
        if value["artifact_set_fingerprint"] != _frozen_binding_fingerprint(value):
            raise TeachObsLockboxError("transitive frozen-model fingerprint mismatch")
    else:
        if (
            any(value[field] is not None for field in digest_fields)
            or any(
                arm_value[field] is not None
                for arm_value in checked_arm_values
                for field in (
                    "manifest_file_sha256",
                    "manifest_sha256",
                    "arrays_file_sha256",
                    "model_thresholds_sha256",
                )
            )
            or any(
                arm_value["model_threshold_count"] is not None
                for arm_value in checked_arm_values
            )
            or value["companion_arrays_loaded_with_allow_pickle_false"] is not False
            or value["artifact_set_fingerprint"] is not None
        ):
            raise TeachObsLockboxError(
                "pending frozen binding cannot carry unverified companion commitments"
            )
    return bound


def preregistration_fingerprint(preregistration: Mapping[str, Any]) -> str:
    """Fingerprint every preregistration field except the fingerprint itself."""

    if not isinstance(preregistration, Mapping):
        raise TeachObsLockboxError("preregistration must be one JSON object")
    payload = dict(preregistration)
    payload.pop("preregistration_fingerprint", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def teachobs_analysis_plan_fingerprint(
    preregistration: Mapping[str, Any],
) -> str:
    """Return the digest expected in final external-evidence SAP bindings."""

    plan = preregistration.get("analysis_plan")
    if not isinstance(plan, Mapping):
        raise TeachObsLockboxError("preregistration analysis_plan is missing")
    return hashlib.sha256(_canonical_bytes(plan)).hexdigest()


def build_teachobs_lockbox_preregistration(
    *,
    study_id: str,
    system_artifact_path: str | Path | None = None,
    analysis_code_path: str | Path | None = None,
    arm_model_artifact_paths: Mapping[str, str | Path] | None = None,
    target_cluster_field: str = "classroom_id",
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a no-outcome preregistration, optionally binding frozen artifacts.

    Omitting any artifact is allowed only so a project can expose an honest
    pending draft.  Such a draft is valid to inspect but is not execution-ready
    and cannot be externally signed by :func:`sign_teachobs_lockbox_preregistration`.
    """

    _require_id("study_id", study_id)
    if target_cluster_field not in {"classroom_id", "session_id"}:
        raise TeachObsLockboxError(
            "target_cluster_field must be classroom_id or session_id"
        )
    timestamp = created_at_utc or _utc_now()
    _parse_utc("created_at_utc", timestamp)
    raw_models = dict(arm_model_artifact_paths or {})
    unknown_models = set(raw_models) - set(ARM_ORDER)
    if unknown_models:
        raise TeachObsLockboxError(
            f"unknown arm model artifact names: {sorted(unknown_models)}"
        )

    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        definition = ARM_DEFINITIONS[arm]
        arms[arm] = {
            "modalities": list(definition["modalities"]),
            "feature_blocks": list(definition["feature_blocks"]),
            "model_artifact": _artifact_binding(
                f"{arm}_frozen_model", raw_models.get(arm)
            ),
        }
    all_arm_models_bound = all(
        arms[arm]["model_artifact"]["bound"] is True for arm in ARM_ORDER
    )
    system_binding = _artifact_binding("frozen_four_arm_system", system_artifact_path)
    analysis_binding = _artifact_binding(
        "paired_cluster_bootstrap_analysis_code", analysis_code_path
    )
    transitive_model_binding = _pending_frozen_model_binding()
    if system_binding["bound"] and all_arm_models_bound:
        try:
            transitive_model_binding = _verified_frozen_model_binding(
                system_artifact_path, raw_models
            )
        except TeachObsLockboxError as exc:
            raise TeachObsLockboxError(
                f"invalid transitive frozen-model artifact set: {exc}"
            ) from exc
    artifacts_complete = bool(
        system_binding["bound"]
        and analysis_binding["bound"]
        and all_arm_models_bound
        and transitive_model_binding["bound"]
    )

    value: dict[str, Any] = {
        "schema_version": "1.2",
        "protocol": PROTOCOL,
        "study_id": study_id,
        "created_at_utc": timestamp,
        "scope": {
            "development_dataset": "TeachObs_v0.1_public_official_23_7_split",
            "development_test_labels_accessible_before_system_freeze": True,
            "development_split_eligible_as_confirmatory_lockbox": False,
            "required_confirmation_context": "new_site_real_classroom",
            "valid_claim_if_completed": "confirmatory_multimodal_gain",
            "deployment_accuracy_claimed_by_this_protocol": False,
            "learner_effectiveness_claimed_by_this_protocol": False,
        },
        "frozen_artifacts": {
            "system_artifact": system_binding,
            "analysis_code_artifact": analysis_binding,
            "arms": arms,
            "transitive_model_binding": transitive_model_binding,
            "all_four_arm_models_bound": all_arm_models_bound,
            "frozen_artifact_set_complete": artifacts_complete,
        },
        "input_contract": {
            "scene_unit": "fixed_15_second_scene_or_preregistered_equivalent",
            "transcript_source": "official_caption_or_media_bound_audited_asr",
            "audio_and_visual_bound_to_same_media_sha256": True,
            "ocr_frames_time_aligned_to_scene": True,
            "feature_extraction_frozen_before_target_outcome_access": True,
            "target_labels_never_used_for_feature_fitting_or_threshold_selection": True,
        },
        "analysis_plan": {
            "task": "39_label_multilabel_teaching_behavior_recognition",
            "arm_order": list(ARM_ORDER),
            "same_eligible_samples_scored_by_every_arm": True,
            "primary_metric": "macro_f1",
            "primary_metric_definition": "unweighted_mean_of_39_per_label_f1_values",
            "primary_comparison": dict(PRIMARY_COMPARISON),
            "secondary_comparisons": [dict(item) for item in SECONDARY_COMPARISONS],
            "decision_threshold_policy": _decision_threshold_policy(
                transitive_model_binding
            ),
            "threshold_and_hyperparameters_frozen_before_target_outcome_access": True,
            "minimum_confirmatory_gain": 0.02,
            "alpha": 0.05,
            "paired_cluster_bootstrap": {
                "paired_resampling": True,
                "cluster_field": target_cluster_field,
                "cluster_unit_definition": (
                    "one complete classroom lesson/session; every scene from a "
                    "cluster is resampled together"
                ),
                "replicates": 2000,
                "seed": 20260723,
                "confidence_level": 0.95,
                "interval_method": "percentile",
            },
            "significance_test": {
                "method": "paired_cluster_bootstrap_tail_with_plus_one_correction",
                "null_delta": "minimum_confirmatory_gain",
                "alternative": "gain_greater_than_registered_minimum",
                "formula": (
                    "(1 + count(bootstrap_delta <= minimum_confirmatory_gain)) "
                    "/ (replicates + 1)"
                ),
                "uses_same_2000_paired_cluster_replicates": True,
            },
            "multiplicity_policy": (
                "only full_minus_transcript_only is confirmatory; all other "
                "four-arm contrasts are secondary"
            ),
        },
        "target_lockbox_protocol": {
            "new_site_identity_disjoint_from_all_development_data_required": True,
            "prospective_collection_required": True,
            "labels_held_by_external_custodian_until_freeze_required": True,
            "maximum_primary_evaluations": 1,
            "minimum_independent_clusters": 10,
            "minimum_coverage_fraction": 0.8,
            "coverage_denominator_defined_before_outcome_access": True,
            "exclusions_must_be_preregistered_and_count_reconciled": True,
            "one_time_consumption_ledger_required": True,
            "public_teachobs_23_7_explicitly_excluded": True,
        },
        "external_governance_gate": {
            "status": "pending_external_registration",
            "registry_name": None,
            "registration_id": None,
            "trusted_public_key_sha256": None,
            "registration_attestation_sha256": None,
            "registration_signature_verified": False,
            "registered_before_target_outcome_access_verified": False,
            "independent_external_governance_verified": False,
        },
        "execution_evidence": {
            "target_dataset_fingerprint": None,
            "target_feature_bundle_fingerprint": None,
            "eligible_sample_set_sha256": None,
            "evaluated_sample_order_sha256": None,
            "cluster_assignment_sha256": None,
            "development_overlap_audit_sha256": None,
            "known_development_overlap_count": None,
            "coverage_evidence_sha256": None,
            "paired_prediction_table_sha256": None,
            "statistical_output_sha256": None,
            "eligible_unit_count": None,
            "evaluated_unit_count": None,
            "actual_independent_cluster_count": None,
            "collection_started_at_utc": None,
            "labels_unsealed_at_utc": None,
            "one_time_consumption_receipt_sha256": None,
            "external_result_attestation_sha256": None,
            "results_computed": False,
        },
        "claim_status": {
            "preregistration_execution_ready": False,
            "external_lockbox_established": False,
            "confirmatory_multimodal_gain_established": False,
            "deployment_accuracy_established": False,
            "learner_effectiveness_established": False,
            "reason": (
                "A preregistration draft contains no target outcomes and no "
                "independent external registration/result signatures."
            ),
        },
    }
    value["preregistration_fingerprint"] = preregistration_fingerprint(value)
    validate_teachobs_lockbox_preregistration(value)
    return value


def validate_teachobs_lockbox_preregistration(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact no-outcome protocol and recompute readiness gates."""

    root = _require_exact_keys(
        "preregistration",
        preregistration,
        {
            "schema_version",
            "protocol",
            "study_id",
            "created_at_utc",
            "scope",
            "frozen_artifacts",
            "input_contract",
            "analysis_plan",
            "target_lockbox_protocol",
            "external_governance_gate",
            "execution_evidence",
            "claim_status",
            "preregistration_fingerprint",
        },
    )
    if root["schema_version"] != "1.2" or root["protocol"] != PROTOCOL:
        raise TeachObsLockboxError("unsupported TeachObs lockbox protocol")
    _require_id("study_id", root["study_id"])
    _parse_utc("created_at_utc", root["created_at_utc"])

    scope = _require_exact_keys(
        "scope",
        root["scope"],
        {
            "development_dataset",
            "development_test_labels_accessible_before_system_freeze",
            "development_split_eligible_as_confirmatory_lockbox",
            "required_confirmation_context",
            "valid_claim_if_completed",
            "deployment_accuracy_claimed_by_this_protocol",
            "learner_effectiveness_claimed_by_this_protocol",
        },
    )
    expected_scope = {
        "development_dataset": "TeachObs_v0.1_public_official_23_7_split",
        "development_test_labels_accessible_before_system_freeze": True,
        "development_split_eligible_as_confirmatory_lockbox": False,
        "required_confirmation_context": "new_site_real_classroom",
        "valid_claim_if_completed": "confirmatory_multimodal_gain",
        "deployment_accuracy_claimed_by_this_protocol": False,
        "learner_effectiveness_claimed_by_this_protocol": False,
    }
    if scope != expected_scope:
        raise TeachObsLockboxError(
            "scope must explicitly exclude public TeachObs 23/7 from lockbox claims"
        )

    artifacts = _require_exact_keys(
        "frozen_artifacts",
        root["frozen_artifacts"],
        {
            "system_artifact",
            "analysis_code_artifact",
            "arms",
            "transitive_model_binding",
            "all_four_arm_models_bound",
            "frozen_artifact_set_complete",
        },
    )
    system_bound = _binding_is_complete(
        "frozen_artifacts.system_artifact", artifacts["system_artifact"]
    )
    analysis_bound = _binding_is_complete(
        "frozen_artifacts.analysis_code_artifact",
        artifacts["analysis_code_artifact"],
    )
    arms = _require_exact_keys(
        "frozen_artifacts.arms", artifacts["arms"], set(ARM_ORDER)
    )
    arm_model_bound: dict[str, bool] = {}
    for arm in ARM_ORDER:
        arm_value = _require_exact_keys(
            f"frozen_artifacts.arms.{arm}",
            arms[arm],
            {"modalities", "feature_blocks", "model_artifact"},
        )
        expected = ARM_DEFINITIONS[arm]
        if arm_value["modalities"] != expected["modalities"]:
            raise TeachObsLockboxError(f"{arm} modalities differ from frozen protocol")
        if arm_value["feature_blocks"] != expected["feature_blocks"]:
            raise TeachObsLockboxError(
                f"{arm} feature blocks differ from frozen protocol"
            )
        arm_model_bound[arm] = _binding_is_complete(
            f"frozen_artifacts.arms.{arm}.model_artifact",
            arm_value["model_artifact"],
        )
    all_models_bound = all(arm_model_bound.values())
    if all_models_bound:
        model_digests = {
            arms[arm]["model_artifact"]["sha256"] for arm in ARM_ORDER
        }
        if len(model_digests) != len(ARM_ORDER):
            raise TeachObsLockboxError(
                "each arm must bind a distinct serialized model artifact"
            )
    if artifacts["all_four_arm_models_bound"] is not all_models_bound:
        raise TeachObsLockboxError("all_four_arm_models_bound is not recomputable")
    transitive_model_bound = _validate_frozen_model_binding(
        artifacts["transitive_model_binding"]
    )
    if transitive_model_bound is not (system_bound and all_models_bound):
        raise TeachObsLockboxError(
            "transitive frozen-model binding differs from supplied artifact bindings"
        )
    if transitive_model_bound:
        transitive = artifacts["transitive_model_binding"]
        if (
            transitive["bundle_manifest_file_sha256"]
            != artifacts["system_artifact"]["sha256"]
            or any(
                transitive["arms"][arm]["manifest_file_sha256"]
                != arms[arm]["model_artifact"]["sha256"]
                for arm in ARM_ORDER
            )
        ):
            raise TeachObsLockboxError(
                "manifest file bindings differ from transitive bundle commitments"
            )
    artifact_set_complete = (
        system_bound and analysis_bound and all_models_bound and transitive_model_bound
    )
    if artifacts["frozen_artifact_set_complete"] is not artifact_set_complete:
        raise TeachObsLockboxError("frozen_artifact_set_complete is not recomputable")

    input_contract = _require_exact_keys(
        "input_contract",
        root["input_contract"],
        {
            "scene_unit",
            "transcript_source",
            "audio_and_visual_bound_to_same_media_sha256",
            "ocr_frames_time_aligned_to_scene",
            "feature_extraction_frozen_before_target_outcome_access",
            "target_labels_never_used_for_feature_fitting_or_threshold_selection",
        },
    )
    expected_input_contract = {
        "scene_unit": "fixed_15_second_scene_or_preregistered_equivalent",
        "transcript_source": "official_caption_or_media_bound_audited_asr",
        "audio_and_visual_bound_to_same_media_sha256": True,
        "ocr_frames_time_aligned_to_scene": True,
        "feature_extraction_frozen_before_target_outcome_access": True,
        "target_labels_never_used_for_feature_fitting_or_threshold_selection": True,
    }
    if input_contract != expected_input_contract:
        raise TeachObsLockboxError("input_contract is incomplete or weakened")

    plan = _require_exact_keys(
        "analysis_plan",
        root["analysis_plan"],
        {
            "task",
            "arm_order",
            "same_eligible_samples_scored_by_every_arm",
            "primary_metric",
            "primary_metric_definition",
            "primary_comparison",
            "secondary_comparisons",
            "decision_threshold_policy",
            "threshold_and_hyperparameters_frozen_before_target_outcome_access",
            "minimum_confirmatory_gain",
            "alpha",
            "paired_cluster_bootstrap",
            "significance_test",
            "multiplicity_policy",
        },
    )
    if plan["task"] != "39_label_multilabel_teaching_behavior_recognition":
        raise TeachObsLockboxError("analysis task differs from the frozen protocol")
    if plan["arm_order"] != list(ARM_ORDER):
        raise TeachObsLockboxError("analysis must contain the exact four-arm order")
    if plan["same_eligible_samples_scored_by_every_arm"] is not True:
        raise TeachObsLockboxError("all four arms must score the same eligible samples")
    if plan["primary_metric"] != "macro_f1" or plan[
        "primary_metric_definition"
    ] != "unweighted_mean_of_39_per_label_f1_values":
        raise TeachObsLockboxError("the primary metric must be fixed 39-label Macro-F1")
    if plan["primary_comparison"] != PRIMARY_COMPARISON:
        raise TeachObsLockboxError("primary contrast must be full minus transcript-only")
    if plan["secondary_comparisons"] != SECONDARY_COMPARISONS:
        raise TeachObsLockboxError("secondary four-arm contrasts differ from protocol")
    _validate_decision_threshold_policy(
        plan["decision_threshold_policy"],
        transitive_model_binding=artifacts["transitive_model_binding"],
        transitive_model_bound=transitive_model_bound,
    )
    if plan[
        "threshold_and_hyperparameters_frozen_before_target_outcome_access"
    ] is not True:
        raise TeachObsLockboxError("target outcomes cannot tune thresholds or models")
    minimum_gain = _require_probability(
        "analysis_plan.minimum_confirmatory_gain",
        plan["minimum_confirmatory_gain"],
    )
    if minimum_gain <= 0.0:
        raise TeachObsLockboxError("minimum_confirmatory_gain must be positive")
    alpha = _require_probability("analysis_plan.alpha", plan["alpha"])
    if not 0.0 < alpha <= 0.05:
        raise TeachObsLockboxError("analysis_plan.alpha must be in (0, 0.05]")
    if plan["multiplicity_policy"] != (
        "only full_minus_transcript_only is confirmatory; all other "
        "four-arm contrasts are secondary"
    ):
        raise TeachObsLockboxError("multiplicity policy is incomplete")
    bootstrap = _require_exact_keys(
        "analysis_plan.paired_cluster_bootstrap",
        plan["paired_cluster_bootstrap"],
        {
            "paired_resampling",
            "cluster_field",
            "cluster_unit_definition",
            "replicates",
            "seed",
            "confidence_level",
            "interval_method",
        },
    )
    if bootstrap["paired_resampling"] is not True:
        raise TeachObsLockboxError("bootstrap must retain paired same-sample predictions")
    if bootstrap["cluster_field"] not in {"classroom_id", "session_id"}:
        raise TeachObsLockboxError("bootstrap cluster must be classroom_id or session_id")
    if bootstrap["cluster_unit_definition"] != (
        "one complete classroom lesson/session; every scene from a cluster is "
        "resampled together"
    ):
        raise TeachObsLockboxError("bootstrap cluster definition differs from protocol")
    if bootstrap["replicates"] != 2000 or bootstrap["seed"] != 20260723:
        raise TeachObsLockboxError("bootstrap must use the fixed 2,000-replicate plan")
    if bootstrap["confidence_level"] != 0.95 or bootstrap[
        "interval_method"
    ] != "percentile":
        raise TeachObsLockboxError("bootstrap interval protocol differs from plan")
    significance = _require_exact_keys(
        "analysis_plan.significance_test",
        plan["significance_test"],
        {
            "method",
            "null_delta",
            "alternative",
            "formula",
            "uses_same_2000_paired_cluster_replicates",
        },
    )
    expected_significance = {
        "method": "paired_cluster_bootstrap_tail_with_plus_one_correction",
        "null_delta": "minimum_confirmatory_gain",
        "alternative": "gain_greater_than_registered_minimum",
        "formula": (
            "(1 + count(bootstrap_delta <= minimum_confirmatory_gain)) / "
            "(replicates + 1)"
        ),
        "uses_same_2000_paired_cluster_replicates": True,
    }
    if significance != expected_significance:
        raise TeachObsLockboxError("significance test differs from the frozen plan")

    target = _require_exact_keys(
        "target_lockbox_protocol",
        root["target_lockbox_protocol"],
        {
            "new_site_identity_disjoint_from_all_development_data_required",
            "prospective_collection_required",
            "labels_held_by_external_custodian_until_freeze_required",
            "maximum_primary_evaluations",
            "minimum_independent_clusters",
            "minimum_coverage_fraction",
            "coverage_denominator_defined_before_outcome_access",
            "exclusions_must_be_preregistered_and_count_reconciled",
            "one_time_consumption_ledger_required",
            "public_teachobs_23_7_explicitly_excluded",
        },
    )
    expected_true_fields = (
        "new_site_identity_disjoint_from_all_development_data_required",
        "prospective_collection_required",
        "labels_held_by_external_custodian_until_freeze_required",
        "coverage_denominator_defined_before_outcome_access",
        "exclusions_must_be_preregistered_and_count_reconciled",
        "one_time_consumption_ledger_required",
        "public_teachobs_23_7_explicitly_excluded",
    )
    if any(target[field] is not True for field in expected_true_fields):
        raise TeachObsLockboxError("target lockbox requirements are incomplete")
    if target["maximum_primary_evaluations"] != 1:
        raise TeachObsLockboxError("target lockbox can be evaluated only once")
    if target["minimum_independent_clusters"] < 10:
        raise TeachObsLockboxError("at least ten independent clusters are required")
    coverage = _require_probability(
        "target_lockbox_protocol.minimum_coverage_fraction",
        target["minimum_coverage_fraction"],
    )
    if coverage < 0.8:
        raise TeachObsLockboxError("minimum target coverage cannot be below 0.8")

    governance = _require_exact_keys(
        "external_governance_gate",
        root["external_governance_gate"],
        {
            "status",
            "registry_name",
            "registration_id",
            "trusted_public_key_sha256",
            "registration_attestation_sha256",
            "registration_signature_verified",
            "registered_before_target_outcome_access_verified",
            "independent_external_governance_verified",
        },
    )
    expected_pending_governance = {
        "status": "pending_external_registration",
        "registry_name": None,
        "registration_id": None,
        "trusted_public_key_sha256": None,
        "registration_attestation_sha256": None,
        "registration_signature_verified": False,
        "registered_before_target_outcome_access_verified": False,
        "independent_external_governance_verified": False,
    }
    if governance != expected_pending_governance:
        raise TeachObsLockboxError(
            "the unsigned preregistration must keep every governance gate pending"
        )

    execution = _require_exact_keys(
        "execution_evidence",
        root["execution_evidence"],
        {
            "target_dataset_fingerprint",
            "target_feature_bundle_fingerprint",
            "eligible_sample_set_sha256",
            "evaluated_sample_order_sha256",
            "cluster_assignment_sha256",
            "development_overlap_audit_sha256",
            "known_development_overlap_count",
            "coverage_evidence_sha256",
            "paired_prediction_table_sha256",
            "statistical_output_sha256",
            "eligible_unit_count",
            "evaluated_unit_count",
            "actual_independent_cluster_count",
            "collection_started_at_utc",
            "labels_unsealed_at_utc",
            "one_time_consumption_receipt_sha256",
            "external_result_attestation_sha256",
            "results_computed",
        },
    )
    if execution["results_computed"] is not False or any(
        execution[field] is not None
        for field in execution
        if field != "results_computed"
    ):
        raise TeachObsLockboxError(
            "a preregistration must not contain target outcomes or execution evidence"
        )

    claim = _require_exact_keys(
        "claim_status",
        root["claim_status"],
        {
            "preregistration_execution_ready",
            "external_lockbox_established",
            "confirmatory_multimodal_gain_established",
            "deployment_accuracy_established",
            "learner_effectiveness_established",
            "reason",
        },
    )
    for field in (
        "preregistration_execution_ready",
        "external_lockbox_established",
        "confirmatory_multimodal_gain_established",
        "deployment_accuracy_established",
        "learner_effectiveness_established",
    ):
        if claim[field] is not False:
            raise TeachObsLockboxError(
                f"an unsigned no-outcome preregistration requires {field}=false"
            )
    if claim["reason"] != (
        "A preregistration draft contains no target outcomes and no independent "
        "external registration/result signatures."
    ):
        raise TeachObsLockboxError("claim_status reason differs from protocol")

    expected_fingerprint = preregistration_fingerprint(root)
    if root["preregistration_fingerprint"] != expected_fingerprint:
        raise TeachObsLockboxError("preregistration fingerprint mismatch")
    return {
        "valid": True,
        "study_id": root["study_id"],
        "preregistration_fingerprint": expected_fingerprint,
        "gates": {
            "public_teachobs_excluded_from_lockbox": True,
            "exact_four_arm_protocol": True,
            "same_sample_paired_primary_comparison": True,
            "fixed_2000_replicate_cluster_bootstrap": True,
            "new_site_prospective_one_time_protocol": True,
            "system_artifact_bound": system_bound,
            "analysis_code_bound": analysis_bound,
            "all_four_arm_models_bound": all_models_bound,
            "transitive_model_and_companion_arrays_bound": transitive_model_bound,
            "per_arm_label_thresholds_bound": transitive_model_bound,
            "external_registration_signature_verified": False,
            "target_execution_evidence_complete": False,
            "external_result_signature_verified": False,
        },
        "frozen_artifact_set_complete": artifact_set_complete,
        "preregistration_execution_ready": False,
        "confirmatory_multimodal_gain_established": False,
        "deployment_accuracy_established": False,
        "learner_effectiveness_established": False,
    }


def verify_teachobs_lockbox_artifact_files(
    preregistration: Mapping[str, Any],
    *,
    system_artifact_path: str | Path,
    analysis_code_path: str | Path,
    arm_model_artifact_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Re-hash local frozen files and compare them with a preregistration."""

    report = validate_teachobs_lockbox_preregistration(preregistration)
    raw_models = dict(arm_model_artifact_paths)
    if set(raw_models) != set(ARM_ORDER):
        raise TeachObsLockboxError(
            "artifact verification requires exactly all four arm model files"
        )
    supplied = {
        "system_artifact": _artifact_binding(
            "frozen_four_arm_system", system_artifact_path
        ),
        "analysis_code_artifact": _artifact_binding(
            "paired_cluster_bootstrap_analysis_code", analysis_code_path
        ),
        "arms": {
            arm: _artifact_binding(f"{arm}_frozen_model", raw_models[arm])
            for arm in ARM_ORDER
        },
    }
    expected = preregistration["frozen_artifacts"]
    if supplied["system_artifact"] != expected["system_artifact"]:
        raise TeachObsLockboxError("frozen system artifact hash/size mismatch")
    if supplied["analysis_code_artifact"] != expected["analysis_code_artifact"]:
        raise TeachObsLockboxError("analysis-code artifact hash/size mismatch")
    for arm in ARM_ORDER:
        if supplied["arms"][arm] != expected["arms"][arm]["model_artifact"]:
            raise TeachObsLockboxError(f"{arm} model artifact hash/size mismatch")
    if report["frozen_artifact_set_complete"] is not True:
        raise TeachObsLockboxError("preregistration does not bind a complete artifact set")
    try:
        transitive = _verified_frozen_model_binding(
            system_artifact_path, raw_models
        )
    except TeachObsLockboxError as exc:
        raise TeachObsLockboxError(
            f"frozen model companion verification failed: {exc}"
        ) from exc
    if transitive != expected["transitive_model_binding"]:
        raise TeachObsLockboxError(
            "frozen bundle/manifest/arrays transitive binding mismatch"
        )
    return {
        "verified": True,
        "preregistration_fingerprint": report["preregistration_fingerprint"],
        "system_artifact_sha256": supplied["system_artifact"]["sha256"],
        "analysis_code_sha256": supplied["analysis_code_artifact"]["sha256"],
        "arm_model_sha256": {
            arm: supplied["arms"][arm]["sha256"] for arm in ARM_ORDER
        },
        "arm_numeric_state_sha256": {
            arm: transitive["arms"][arm]["arrays_file_sha256"]
            for arm in ARM_ORDER
        },
        "arm_model_thresholds_sha256": {
            arm: transitive["arms"][arm]["model_thresholds_sha256"]
            for arm in ARM_ORDER
        },
        "decision_threshold_count_per_arm": _LABEL_COUNT,
        "artifact_set_fingerprint": transitive["artifact_set_fingerprint"],
        "transitive_model_and_companion_arrays_verified": True,
        "paths_included": False,
    }


def validate_teachobs_confirmatory_evidence_handoff(
    preregistration: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a completed generic external-evidence manifest back to this plan.

    Signature verification remains the responsibility of
    :func:`external_evidence.verify_external_research_evidence_attestation`.
    This function closes the semantic gap between that generic two-system
    aggregate manifest and the exact TeachObs-style four-arm preregistration.
    """

    prereg_report = validate_teachobs_lockbox_preregistration(preregistration)
    if prereg_report["frozen_artifact_set_complete"] is not True:
        raise TeachObsLockboxError(
            "confirmatory evidence cannot bind an incomplete four-arm artifact set"
        )
    try:
        from .external_evidence import (
            ExternalEvidenceError,
            validate_external_research_evidence,
        )

        evidence_report = validate_external_research_evidence(
            evidence,
            expected_kind="confirmatory_multimodal_gain",
        )
    except ExternalEvidenceError as exc:
        raise TeachObsLockboxError(
            f"generic external evidence is invalid: {exc}"
        ) from exc
    prereg_evidence = evidence.get("preregistration")
    bindings = evidence.get("artifact_bindings")
    claim = evidence.get("claim")
    population = evidence.get("population")
    if not all(
        isinstance(value, Mapping)
        for value in (prereg_evidence, bindings, claim, population)
    ):
        raise TeachObsLockboxError("external evidence handoff sections are missing")
    expected_protocol_sha256 = prereg_report["preregistration_fingerprint"]
    if prereg_evidence.get("protocol_sha256") != expected_protocol_sha256:
        raise TeachObsLockboxError(
            "external evidence protocol_sha256 does not bind this preregistration"
        )
    expected_sap_sha256 = teachobs_analysis_plan_fingerprint(preregistration)
    if (
        prereg_evidence.get("statistical_analysis_plan_sha256")
        != expected_sap_sha256
    ):
        raise TeachObsLockboxError(
            "external evidence statistical plan does not bind the frozen four-arm plan"
        )
    artifacts = preregistration["frozen_artifacts"]
    expected_system_sha256 = artifacts["system_artifact"]["sha256"]
    expected_comparator_sha256 = artifacts["arms"]["transcript_only"][
        "model_artifact"
    ]["sha256"]
    expected_analysis_sha256 = artifacts["analysis_code_artifact"]["sha256"]
    expected_bindings = {
        "evaluated_system_fingerprint": expected_system_sha256,
        "comparator_fingerprint": expected_comparator_sha256,
        "analysis_code_sha256": expected_analysis_sha256,
    }
    for name, expected in expected_bindings.items():
        if bindings.get(name) != expected:
            raise TeachObsLockboxError(
                f"external evidence artifact binding mismatch: {name}"
            )
    expected_claim_fields = {
        "evaluation_design": "paired_same_samples_external_lockbox",
        "baseline_modalities": ["transcript"],
        "added_modalities": ["audio", "visual"],
        "primary_metric": preregistration["analysis_plan"]["primary_metric"],
        "minimum_confirmatory_gain": preregistration["analysis_plan"][
            "minimum_confirmatory_gain"
        ],
        "alpha": preregistration["analysis_plan"]["alpha"],
        "paired_cluster_aware_inference": True,
        "claim_cluster_field": preregistration["analysis_plan"][
            "paired_cluster_bootstrap"
        ]["cluster_field"],
    }
    for name, expected in expected_claim_fields.items():
        if claim.get(name) != expected:
            raise TeachObsLockboxError(
                f"external evidence claim differs from preregistration: {name}"
            )
    minimum_clusters = preregistration["target_lockbox_protocol"][
        "minimum_independent_clusters"
    ]
    if claim.get("claim_cluster_count", 0) < minimum_clusters:
        raise TeachObsLockboxError(
            "external evidence has fewer clusters than the preregistered minimum"
        )
    minimum_coverage = preregistration["target_lockbox_protocol"][
        "minimum_coverage_fraction"
    ]
    if population.get("coverage_fraction", 0.0) < minimum_coverage:
        raise TeachObsLockboxError(
            "external evidence coverage is below the preregistered minimum"
        )
    if population.get("identity_disjoint_from_development") is not True:
        raise TeachObsLockboxError(
            "external evidence is not identity-disjoint from development"
        )
    return {
        "valid": True,
        "handoff_verified": True,
        "preregistration_fingerprint": expected_protocol_sha256,
        "statistical_analysis_plan_sha256": expected_sap_sha256,
        "four_arm_system_fingerprint": expected_system_sha256,
        "transcript_only_comparator_fingerprint": expected_comparator_sha256,
        "analysis_code_sha256": expected_analysis_sha256,
        "same_sample_paired_claim_bound": True,
        "fixed_cluster_bootstrap_plan_bound": True,
        "unsigned_external_evidence_claim_gates_passed": bool(
            evidence_report["claim_established"]
        ),
        "external_evidence_signature_verified": False,
        "signature_still_required": True,
        "confirmatory_multimodal_gain_established": False,
        "deployment_accuracy_established": False,
        "learner_effectiveness_established": False,
    }


def sign_teachobs_lockbox_preregistration(
    preregistration: Mapping[str, Any],
    *,
    private_key_pem: bytes,
    issuer: str,
    key_id: str,
) -> dict[str, Any]:
    """Sign a complete preregistration as its external lockbox custodian.

    A developer can exercise this function in a test, but a local self-signature
    does not establish signer independence.  Production verification must use a
    public key provisioned out of band by the external governance party.
    """

    report = validate_teachobs_lockbox_preregistration(preregistration)
    if report["frozen_artifact_set_complete"] is not True:
        raise TeachObsLockboxError(
            "external registration requires system, analysis code, and all four models"
        )
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 lockbox registration requires `pip install -e '.[recognition]'`"
        ) from exc
    if not isinstance(issuer, str) or not issuer.strip():
        raise TeachObsLockboxError("issuer is required")
    _require_id("key_id", key_id)
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TeachObsLockboxError("private key must be Ed25519")
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    statement = {
        "schema_version": "1.0",
        "protocol": "teachobs_new_site_preregistration_signature_statement_v1",
        "issuer": issuer.strip(),
        "key_id": key_id,
        "signer_role": "independent_external_lockbox_governance",
        "signed_at_utc": _utc_now(),
        "custodian_attestations": {
            "new_site_identity_disjoint_from_development": True,
            "prospective_collection": True,
            "outcomes_unavailable_to_system_developers_at_registration": True,
            "labels_not_used_to_modify_models_features_thresholds_or_claims": True,
            "public_teachobs_23_7_not_used_as_lockbox": True,
            "maximum_primary_evaluations": 1,
        },
        "preregistration": dict(preregistration),
    }
    signature = private_key.sign(_canonical_bytes(statement))
    return {
        "schema_version": "1.0",
        "protocol": ATTESTATION_PROTOCOL,
        "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
        "statement": statement,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_teachobs_lockbox_preregistration_attestation(
    attestation: Mapping[str, Any],
    *,
    trusted_public_key_pem: bytes,
    expected_preregistration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify external signature, exact preregistration, and custodian gates."""

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 lockbox registration requires `pip install -e '.[recognition]'`"
        ) from exc
    outer = _require_exact_keys(
        "preregistration attestation",
        attestation,
        {
            "schema_version",
            "protocol",
            "public_key_sha256",
            "statement",
            "signature_base64",
        },
    )
    if outer["schema_version"] != "1.0" or outer["protocol"] != ATTESTATION_PROTOCOL:
        raise TeachObsLockboxError("unsupported preregistration attestation protocol")
    statement = _require_exact_keys(
        "preregistration signature statement",
        outer["statement"],
        {
            "schema_version",
            "protocol",
            "issuer",
            "key_id",
            "signer_role",
            "signed_at_utc",
            "custodian_attestations",
            "preregistration",
        },
    )
    if statement["schema_version"] != "1.0" or statement["protocol"] != (
        "teachobs_new_site_preregistration_signature_statement_v1"
    ):
        raise TeachObsLockboxError("unsupported preregistration signature statement")
    if not isinstance(statement["issuer"], str) or not statement["issuer"].strip():
        raise TeachObsLockboxError("signature issuer is missing")
    _require_id("statement.key_id", statement["key_id"])
    if statement["signer_role"] != "independent_external_lockbox_governance":
        raise TeachObsLockboxError("signature role is not external lockbox governance")
    signed_at = _parse_utc("statement.signed_at_utc", statement["signed_at_utc"])
    expected_custodian_attestations = {
        "new_site_identity_disjoint_from_development": True,
        "prospective_collection": True,
        "outcomes_unavailable_to_system_developers_at_registration": True,
        "labels_not_used_to_modify_models_features_thresholds_or_claims": True,
        "public_teachobs_23_7_not_used_as_lockbox": True,
        "maximum_primary_evaluations": 1,
    }
    if statement["custodian_attestations"] != expected_custodian_attestations:
        raise TeachObsLockboxError("custodian attestations are incomplete or weakened")
    raw_preregistration = statement["preregistration"]
    if not isinstance(raw_preregistration, Mapping):
        raise TeachObsLockboxError("signed preregistration is missing")
    prereg_report = validate_teachobs_lockbox_preregistration(raw_preregistration)
    created_at = _parse_utc(
        "preregistration.created_at_utc", raw_preregistration["created_at_utc"]
    )
    if signed_at < created_at:
        raise TeachObsLockboxError("preregistration was signed before it was created")
    if prereg_report["frozen_artifact_set_complete"] is not True:
        raise TeachObsLockboxError("signed preregistration lacks frozen artifacts")
    if expected_preregistration is not None and dict(expected_preregistration) != dict(
        raw_preregistration
    ):
        raise TeachObsLockboxError(
            "standalone preregistration differs from the signed preregistration"
        )
    public_key = serialization.load_pem_public_key(trusted_public_key_pem)
    if not isinstance(public_key, Ed25519PublicKey):
        raise TeachObsLockboxError("trusted public key must be Ed25519")
    public_der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_key_sha256 = hashlib.sha256(public_der).hexdigest()
    if outer["public_key_sha256"] != public_key_sha256:
        raise TeachObsLockboxError("trusted public-key fingerprint mismatch")
    try:
        signature = base64.b64decode(str(outer["signature_base64"]), validate=True)
        public_key.verify(signature, _canonical_bytes(statement))
    except (ValueError, InvalidSignature) as exc:
        raise TeachObsLockboxError(
            "preregistration signature verification failed"
        ) from exc
    return {
        "valid": True,
        "signature_verified": True,
        "issuer": statement["issuer"].strip(),
        "key_id": statement["key_id"],
        "public_key_sha256": public_key_sha256,
        "preregistration_fingerprint": prereg_report[
            "preregistration_fingerprint"
        ],
        "external_governance_registration_gate_verified": True,
        "signature_alone_proves_signer_independence": False,
        "target_execution_evidence_complete": False,
        "confirmatory_multimodal_gain_established": False,
        "deployment_accuracy_established": False,
        "learner_effectiveness_established": False,
    }


__all__ = [
    "ARM_DEFINITIONS",
    "ARM_ORDER",
    "TeachObsLockboxError",
    "build_teachobs_lockbox_preregistration",
    "preregistration_fingerprint",
    "sign_teachobs_lockbox_preregistration",
    "teachobs_analysis_plan_fingerprint",
    "validate_teachobs_confirmatory_evidence_handoff",
    "validate_teachobs_lockbox_preregistration",
    "verify_teachobs_lockbox_artifact_files",
    "verify_teachobs_lockbox_preregistration_attestation",
]
