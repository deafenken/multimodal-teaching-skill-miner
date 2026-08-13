"""Privacy-minimizing cluster-randomized learner-effect study tooling.

The module creates a preregistration package and analyzes genuinely completed
tables.  It never creates participant identities or positive learning-effect
evidence.  Local analysis remains explicitly unestablished until the existing
externally governed evidence and signature chain validates the real study.
"""

from __future__ import annotations

import csv
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random
import re
import secrets
import sys
from typing import Any, Iterable

from .io_utils import ensure_private_directory, read_json, write_json, write_text


PREREGISTRATION_SCHEMA = "teaching_skill_miner.learner_effect_preregistration.v2"
ANALYSIS_SCHEMA = "teaching_skill_miner.learner_effect_analysis.v2"
EXTERNAL_ASSESSMENT_ATTESTATION_SCHEMA = (
    "teaching_skill_miner.learner_effect_external_assessment_attestation.v1"
)
PROTOCOL_VERSION = "2.0"
EXTERNAL_ASSESSMENT_PROTOCOL = "ed25519_learner_effect_external_assessment_v1"
STUDY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOCATION_FIELDS = (
    "preregistration_core_sha256",
    "participant_token",
    "cluster_token",
    "assigned_arm",
    "randomized",
)
OUTCOME_FIELDS = (
    "preregistration_core_sha256",
    "participant_token",
    "cluster_token",
    "pre_score",
    "post_score",
    "transfer_score",
    "retention_score",
    "primary_missing_reason",
    "transfer_missing_reason",
    "retention_missing_reason",
    "retention_assessment_day",
    "subgroup_code",
    "grader_blind",
    "adverse_event_reported",
    "adverse_event_severity",
    "adverse_event_relatedness",
    "protocol_deviation",
)
TEACHER_BURDEN_FIELDS = (
    "preregistration_core_sha256",
    "cluster_token",
    "preparation_minutes",
    "delivery_minutes",
    "followup_minutes",
    "workload_rating_1_to_5",
    "burden_missing_reason",
)
MISSING_REASONS = {
    "not_missing",
    "withdrawn",
    "lost_to_followup",
    "illness",
    "technical_failure",
    "other",
}
PROTOCOL_DEVIATIONS = {
    "none",
    "nonadherence",
    "crossover",
    "absence",
    "other",
}
ADVERSE_EVENT_SEVERITIES = {"none", "mild", "moderate", "serious"}
ADVERSE_EVENT_RELATEDNESS = {"none", "unrelated", "possibly", "probably"}
BURDEN_MISSING_REASONS = MISSING_REASONS
SUBGROUP_CODE_RE = re.compile(r"^sg[0-9]{2}$")


class LearnerEffectStudyError(ValueError):
    """Raised when a study package or filled table fails closed validation."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LearnerEffectStudyError(
            "study artifact is not canonical finite JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token(secret: bytes, namespace: str) -> str:
    return hashlib.sha256(secret + b"\0" + namespace.encode("utf-8")).hexdigest()


def _csv_text(fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _validate_generation_parameters(
    *,
    study_id: str,
    cluster_unit: str,
    cluster_count: int,
    participants_per_cluster: int,
    minimum_cluster_count: int,
    minimum_primary_coverage: float,
    maximum_attrition_fraction: float,
    maximum_differential_attrition_fraction: float,
    minimum_meaningful_effect: float,
    score_minimum: float,
    score_maximum: float,
    bootstrap_replicates: int,
    data_origin: str,
    retention_window_days: tuple[int, int],
    subgroup_codes: tuple[str, ...],
    minimum_subgroup_cell_size: int,
    maximum_fairness_effect_gap: float,
) -> None:
    if not STUDY_ID_RE.fullmatch(study_id):
        raise LearnerEffectStudyError("study_id has an unsafe or unsupported format")
    if cluster_unit not in {"teacher", "classroom"}:
        raise LearnerEffectStudyError("cluster_unit must be teacher or classroom")
    if (
        isinstance(cluster_count, bool)
        or not isinstance(cluster_count, int)
        or cluster_count < 2
        or cluster_count > 1000
        or cluster_count % 2
    ):
        raise LearnerEffectStudyError("cluster_count must be an even integer in 2-1000")
    if (
        isinstance(participants_per_cluster, bool)
        or not isinstance(participants_per_cluster, int)
        or not 1 <= participants_per_cluster <= 1000
    ):
        raise LearnerEffectStudyError("participants_per_cluster must be in 1-1000")
    if (
        isinstance(minimum_cluster_count, bool)
        or not isinstance(minimum_cluster_count, int)
        or minimum_cluster_count < 6
        or minimum_cluster_count > cluster_count
    ):
        raise LearnerEffectStudyError(
            "minimum_cluster_count must be preregistered at 6 or more and not exceed plan"
        )
    if not 0.8 <= float(minimum_primary_coverage) <= 1.0:
        raise LearnerEffectStudyError("minimum_primary_coverage must be in 0.8-1.0")
    if not 0.0 <= float(maximum_attrition_fraction) <= 0.2:
        raise LearnerEffectStudyError("maximum_attrition_fraction must be in 0.0-0.2")
    if (
        1.0 - float(minimum_primary_coverage)
        > float(maximum_attrition_fraction) + 1e-12
    ):
        raise LearnerEffectStudyError(
            "coverage and attrition gates are internally inconsistent"
        )
    if not 0.0 <= float(maximum_differential_attrition_fraction) <= 0.2:
        raise LearnerEffectStudyError(
            "maximum_differential_attrition_fraction must be in 0.0-0.2"
        )
    if not 0.0 < float(minimum_meaningful_effect) <= 10.0:
        raise LearnerEffectStudyError("minimum_meaningful_effect must be positive")
    if (
        not math.isfinite(float(score_minimum))
        or not math.isfinite(float(score_maximum))
        or float(score_minimum) >= float(score_maximum)
    ):
        raise LearnerEffectStudyError("score bounds must be finite and increasing")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or not 20 <= bootstrap_replicates <= 100_000
    ):
        raise LearnerEffectStudyError("bootstrap_replicates must be in 20-100000")
    if data_origin not in {"template", "synthetic", "real"}:
        raise LearnerEffectStudyError(
            "data_origin must be template, synthetic, or real"
        )
    if (
        not isinstance(retention_window_days, tuple)
        or len(retention_window_days) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in retention_window_days
        )
        or not 14 <= retention_window_days[0] <= retention_window_days[1] <= 28
    ):
        raise LearnerEffectStudyError(
            "retention_window_days must be an integer (start, end) within 14-28 days"
        )
    if (
        not isinstance(subgroup_codes, tuple)
        or not 2 <= len(subgroup_codes) <= 8
        or len(set(subgroup_codes)) != len(subgroup_codes)
        or any(SUBGROUP_CODE_RE.fullmatch(item) is None for item in subgroup_codes)
    ):
        raise LearnerEffectStudyError(
            "subgroup_codes must contain 2-8 unique privacy-safe codes such as sg01"
        )
    if (
        isinstance(minimum_subgroup_cell_size, bool)
        or not isinstance(minimum_subgroup_cell_size, int)
        or not 5 <= minimum_subgroup_cell_size <= 100
    ):
        raise LearnerEffectStudyError("minimum_subgroup_cell_size must be in 5-100")
    if (
        not math.isfinite(float(maximum_fairness_effect_gap))
        or not 0.0 < float(maximum_fairness_effect_gap) <= 5.0
    ):
        raise LearnerEffectStudyError("maximum_fairness_effect_gap must be positive")


def generate_learner_effect_study_package(
    output_directory: str | Path,
    *,
    study_id: str,
    cluster_unit: str = "classroom",
    cluster_count: int = 6,
    participants_per_cluster: int = 5,
    randomization_seed: int = 20_260_723,
    bootstrap_seed: int = 20_260_724,
    bootstrap_replicates: int = 2_000,
    minimum_cluster_count: int = 6,
    minimum_primary_coverage: float = 0.8,
    maximum_attrition_fraction: float = 0.2,
    maximum_differential_attrition_fraction: float = 0.1,
    minimum_meaningful_effect: float = 0.1,
    score_minimum: float = 0.0,
    score_maximum: float = 100.0,
    intervention_description: str = "teaching-skill-assisted instruction",
    comparator_description: str = "preregistered active teaching comparator",
    data_origin: str = "template",
    ethics_approval_id: str = "",
    informed_consent_or_approved_waiver: bool = False,
    preregistration_frozen_before_allocation: bool = False,
    allocation_concealment_procedure_declared: bool = False,
    intervention_version: str = "unfrozen",
    intervention_build_sha256: str | None = None,
    semantic_lockbox_report_sha256: str | None = None,
    independent_assessor_organization: str = "external_validation_pending",
    independent_assessor_key_id: str = "external_validation_pending",
    assessor_independence_declared: bool = False,
    retention_window_days: tuple[int, int] = (14, 28),
    minimum_transfer_coverage: float = 0.8,
    minimum_retention_coverage: float = 0.8,
    subgroup_codes: tuple[str, ...] = ("sg01", "sg02"),
    subgroup_definitions_registry_sha256: str | None = None,
    minimum_subgroup_cell_size: int = 5,
    maximum_fairness_effect_gap: float = 0.5,
    token_secret: bytes | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create private preregistration, blinded allocation, and outcome templates."""

    _validate_generation_parameters(
        study_id=study_id,
        cluster_unit=cluster_unit,
        cluster_count=cluster_count,
        participants_per_cluster=participants_per_cluster,
        minimum_cluster_count=minimum_cluster_count,
        minimum_primary_coverage=minimum_primary_coverage,
        maximum_attrition_fraction=maximum_attrition_fraction,
        maximum_differential_attrition_fraction=maximum_differential_attrition_fraction,
        minimum_meaningful_effect=minimum_meaningful_effect,
        score_minimum=score_minimum,
        score_maximum=score_maximum,
        bootstrap_replicates=bootstrap_replicates,
        data_origin=data_origin,
        retention_window_days=retention_window_days,
        subgroup_codes=subgroup_codes,
        minimum_subgroup_cell_size=minimum_subgroup_cell_size,
        maximum_fairness_effect_gap=maximum_fairness_effect_gap,
    )
    if (
        not isinstance(randomization_seed, int)
        or isinstance(randomization_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or isinstance(bootstrap_seed, bool)
    ):
        raise LearnerEffectStudyError(
            "randomization and bootstrap seeds must be integers"
        )
    if not intervention_description.strip() or not comparator_description.strip():
        raise LearnerEffectStudyError("both arm descriptions must be non-empty")
    if (
        not isinstance(informed_consent_or_approved_waiver, bool)
        or not isinstance(preregistration_frozen_before_allocation, bool)
        or not isinstance(allocation_concealment_procedure_declared, bool)
    ):
        raise LearnerEffectStudyError("governance declarations must be booleans")
    if not isinstance(assessor_independence_declared, bool):
        raise LearnerEffectStudyError("assessor_independence_declared must be boolean")
    if not intervention_version.strip() or len(intervention_version) > 128:
        raise LearnerEffectStudyError(
            "intervention_version must be non-empty and bounded"
        )
    for value, name in (
        (intervention_build_sha256, "intervention_build_sha256"),
        (semantic_lockbox_report_sha256, "semantic_lockbox_report_sha256"),
        (subgroup_definitions_registry_sha256, "subgroup_definitions_registry_sha256"),
    ):
        if value is not None and SHA256_RE.fullmatch(value) is None:
            raise LearnerEffectStudyError(f"{name} must be a lowercase SHA-256 or null")
    if not 0.8 <= float(minimum_transfer_coverage) <= 1.0:
        raise LearnerEffectStudyError("minimum_transfer_coverage must be in 0.8-1.0")
    if not 0.8 <= float(minimum_retention_coverage) <= 1.0:
        raise LearnerEffectStudyError("minimum_retention_coverage must be in 0.8-1.0")
    secret = token_secret if token_secret is not None else secrets.token_bytes(32)
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise LearnerEffectStudyError(
            "token_secret must contain at least 16 private bytes"
        )
    generated = generated_at_utc or _utc_now()
    core: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "study_id": study_id,
        "generated_at_utc": generated,
        "design": {
            "study_design": "cluster_randomized_controlled_trial",
            "cluster_unit": cluster_unit,
            "arms": {
                "A": intervention_description.strip(),
                "B": comparator_description.strip(),
            },
            "planned_cluster_count": cluster_count,
            "planned_participants_per_cluster": participants_per_cluster,
            "planned_participant_count": cluster_count * participants_per_cluster,
            "balanced_one_to_one_cluster_assignment": True,
            "randomization_seed": randomization_seed,
            "allocation_unit_is_cluster_not_participant": True,
            "intervention_version": intervention_version.strip(),
            "intervention_build_sha256": intervention_build_sha256,
        },
        "governance": {
            "data_origin": data_origin,
            "real_participant_data_declared": data_origin == "real",
            "ethics_approval_id": ethics_approval_id.strip(),
            "informed_consent_or_approved_waiver": informed_consent_or_approved_waiver,
            "preregistration_frozen_before_allocation": (
                preregistration_frozen_before_allocation
            ),
            "allocation_concealment_procedure_declared": (
                allocation_concealment_procedure_declared
            ),
            "external_governance_signature_included": False,
            "independent_assessor": {
                "organization": independent_assessor_organization.strip(),
                "key_id": independent_assessor_key_id.strip(),
                "independent_from_developer_and_intervention_provider_declared": (
                    assessor_independence_declared
                ),
                "identity_and_independence_externally_verified": False,
            },
        },
        "outcomes": {
            "outcome_source": "independent_learner_assessment",
            "primary_outcome": "post_score_adjusted_for_pre_score",
            "primary_model": "participant_level_ols_post_on_arm_and_pre",
            "effect_measure": "adjusted_standardized_mean_difference",
            "standardizer": "sample_standard_deviation_of_all_randomized_pre_scores",
            "pre_score_required_for_every_randomized_participant": True,
            "missing_post_policy": "baseline_carried_forward_post_equals_pre",
            "secondary_outcomes": [
                "transfer_score_adjusted_for_pre_score",
                "retention_score_at_14_to_28_days_adjusted_for_pre_score",
            ],
            "transfer_assessment_uses_unseen_items": True,
            "retention_window_days": list(retention_window_days),
            "score_minimum": float(score_minimum),
            "score_maximum": float(score_maximum),
            "grader_blinding_required": True,
            "internal_skill_score_as_outcome_prohibited": True,
            "assessor_must_be_independent_and_blind": True,
        },
        "analysis_plan": {
            "estimand": "intention_to_treat_by_randomized_cluster_assignment",
            "cluster_bootstrap": "arm_stratified_resample_clusters_with_replacement",
            "confidence_level": 0.95,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "secondary_bootstrap_seeds": {
                "transfer": bootstrap_seed + 1,
                "retention": bootstrap_seed + 2,
            },
            "minimum_valid_bootstrap_fraction": 0.9,
            "minimum_cluster_count": minimum_cluster_count,
            "minimum_participant_count": 30,
            "minimum_primary_outcome_coverage": float(minimum_primary_coverage),
            "minimum_transfer_outcome_coverage": float(minimum_transfer_coverage),
            "minimum_retention_outcome_coverage": float(minimum_retention_coverage),
            "maximum_attrition_fraction": float(maximum_attrition_fraction),
            "maximum_differential_attrition_fraction": float(
                maximum_differential_attrition_fraction
            ),
            "minimum_educationally_meaningful_standardized_effect": float(
                minimum_meaningful_effect
            ),
            "no_test_outcome_tuning": True,
            "all_randomized_tokens_must_be_present_in_outcome_table": True,
            "subgroup_analysis": {
                "predeclared_codes": list(subgroup_codes),
                "definitions_registry_sha256": subgroup_definitions_registry_sha256,
                "minimum_cell_size_per_arm": minimum_subgroup_cell_size,
                "small_cells_suppressed": True,
                "maximum_allowed_standardized_effect_gap": float(
                    maximum_fairness_effect_gap
                ),
                "subgroup_semantics_stay_with_external_site_custodian": True,
            },
            "teacher_burden": {
                "unit": "randomized_cluster",
                "components": [
                    "preparation_minutes",
                    "delivery_minutes",
                    "followup_minutes",
                    "workload_rating_1_to_5",
                ],
                "all_randomized_clusters_required": True,
            },
        },
        "semantic_lockbox_binding": {
            "report_sha256": semantic_lockbox_report_sha256,
            "required_for_external_claim": True,
            "behavioral_lockbox_is_not_learning_effect_evidence": True,
        },
        "privacy": {
            "participant_identity_fields_permitted": False,
            "teacher_or_classroom_identity_fields_permitted": False,
            "only_sha256_tokens_in_tabular_identifiers": True,
            "identity_token_mapping_must_remain_with_authorized_site_custodian": True,
        },
        "claim_boundary": {
            "template_or_local_analysis_is_external_evidence": False,
            "learner_effectiveness_established": False,
            "requires_existing_external_evidence_signature_chain": True,
            "external_validation_status": "external_validation_pending",
        },
        "tokenization": {
            "algorithm": "sha256(private_random_secret || namespace)",
            "private_secret_sha256": hashlib.sha256(secret).hexdigest(),
            "private_secret_included_in_package": False,
        },
    }
    core_sha256 = _canonical_sha256(core)
    assignments = ["A"] * (cluster_count // 2) + ["B"] * (cluster_count // 2)
    random.Random(randomization_seed).shuffle(assignments)
    allocation_rows: list[dict[str, str]] = []
    outcome_rows: list[dict[str, str]] = []
    burden_rows: list[dict[str, str]] = []
    for cluster_index, arm in enumerate(assignments, start=1):
        cluster_token = _token(secret, f"{study_id}:cluster:{cluster_index}")
        burden_rows.append(
            {
                "preregistration_core_sha256": core_sha256,
                "cluster_token": cluster_token,
                "preparation_minutes": "",
                "delivery_minutes": "",
                "followup_minutes": "",
                "workload_rating_1_to_5": "",
                "burden_missing_reason": "",
            }
        )
        for participant_index in range(1, participants_per_cluster + 1):
            participant_token = _token(
                secret,
                f"{study_id}:cluster:{cluster_index}:participant:{participant_index}",
            )
            allocation_rows.append(
                {
                    "preregistration_core_sha256": core_sha256,
                    "participant_token": participant_token,
                    "cluster_token": cluster_token,
                    "assigned_arm": arm,
                    "randomized": "true",
                }
            )
            outcome_rows.append(
                {
                    "preregistration_core_sha256": core_sha256,
                    "participant_token": participant_token,
                    "cluster_token": cluster_token,
                    "pre_score": "",
                    "post_score": "",
                    "transfer_score": "",
                    "retention_score": "",
                    "primary_missing_reason": "",
                    "transfer_missing_reason": "",
                    "retention_missing_reason": "",
                    "retention_assessment_day": "",
                    "subgroup_code": "",
                    "grader_blind": "",
                    "adverse_event_reported": "",
                    "adverse_event_severity": "",
                    "adverse_event_relatedness": "",
                    "protocol_deviation": "",
                }
            )
    allocation_text = _csv_text(ALLOCATION_FIELDS, allocation_rows)
    outcome_text = _csv_text(OUTCOME_FIELDS, outcome_rows)
    burden_text = _csv_text(TEACHER_BURDEN_FIELDS, burden_rows)
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "preregistration_core": core,
        "preregistration_core_sha256": core_sha256,
        "artifact_bindings": {
            "allocation_csv_sha256": hashlib.sha256(
                allocation_text.encode("utf-8")
            ).hexdigest(),
            "outcome_template_csv_sha256": hashlib.sha256(
                outcome_text.encode("utf-8")
            ).hexdigest(),
            "teacher_burden_template_csv_sha256": hashlib.sha256(
                burden_text.encode("utf-8")
            ).hexdigest(),
        },
        "learner_effectiveness_established": False,
    }
    output = ensure_private_directory(output_directory).resolve()
    preregistration_path = write_json(output / "preregistration.json", preregistration)
    allocation_path = write_text(output / "participant_allocation.csv", allocation_text)
    outcome_path = write_text(output / "outcome_collection_template.csv", outcome_text)
    burden_path = write_text(
        output / "teacher_burden_collection_template.csv", burden_text
    )
    return {
        "mode": "learner_effect_study_templates_generated",
        "output_directory": str(output),
        "preregistration_path": str(preregistration_path),
        "allocation_path": str(allocation_path),
        "outcome_template_path": str(outcome_path),
        "teacher_burden_template_path": str(burden_path),
        "preregistration_core_sha256": core_sha256,
        "planned_cluster_count": cluster_count,
        "planned_participant_count": len(allocation_rows),
        "identity_fields_included": False,
        "real_participant_data_generated": False,
        "learner_effectiveness_established": False,
        "external_governance_signature_included": False,
        "external_validation_status": "external_validation_pending",
        "external_validation_pending_requirements": [
            "independent assessor must blind-grade real participant outcomes",
            "independent assessor Ed25519 attestation must bind the exact analysis",
            "semantic lockbox aggregate report hash and signature chain must verify",
            "confirmatory p-value and every preregistered outcome must be externally reported",
            "ethics, consent or waiver, assessor independence, and public-key trust require external audit",
        ],
    }


def _load_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise LearnerEffectStudyError("preregistration is missing or unsafe")
    value = read_json(path)
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "preregistration_core",
        "preregistration_core_sha256",
        "artifact_bindings",
        "learner_effectiveness_established",
    }:
        raise LearnerEffectStudyError("preregistration has an invalid top-level schema")
    if value.get("schema") != PREREGISTRATION_SCHEMA:
        raise LearnerEffectStudyError(
            "unsupported learner-effect preregistration schema"
        )
    core = value.get("preregistration_core")
    if not isinstance(core, dict) or core.get("schema") != PREREGISTRATION_SCHEMA:
        raise LearnerEffectStudyError("preregistration core is malformed")
    core_sha256 = _canonical_sha256(core)
    if value.get("preregistration_core_sha256") != core_sha256:
        raise LearnerEffectStudyError("preregistration core hash mismatch")
    if value.get("learner_effectiveness_established") is not False:
        raise LearnerEffectStudyError(
            "a preregistration template cannot declare learning effectiveness"
        )
    bindings = value.get("artifact_bindings")
    if (
        not isinstance(bindings, dict)
        or set(bindings)
        != {
            "allocation_csv_sha256",
            "outcome_template_csv_sha256",
            "teacher_burden_template_csv_sha256",
        }
        or any(not SHA256_RE.fullmatch(str(item)) for item in bindings.values())
    ):
        raise LearnerEffectStudyError("preregistration artifact bindings are malformed")
    return value, core, core_sha256


def _read_csv_exact(
    path: Path, expected_fields: tuple[str, ...], purpose: str
) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise LearnerEffectStudyError(f"{purpose} CSV is missing or unsafe")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise LearnerEffectStudyError(f"{purpose} CSV is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise LearnerEffectStudyError(
            f"{purpose} CSV header differs from the fixed privacy-safe schema"
        )
    rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise LearnerEffectStudyError(f"{purpose} CSV is empty or malformed")
    return [{key: str(value) for key, value in row.items()} for row in rows]


def _finite_score(value: str, *, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise LearnerEffectStudyError(f"{name} must be a finite numeric score") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise LearnerEffectStudyError(f"{name} is outside preregistered score bounds")
    return number


def _optional_score(
    value: str,
    reason: str,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    text = value.strip()
    missing_reason = reason.strip()
    if text:
        if missing_reason != "not_missing":
            raise LearnerEffectStudyError(
                f"{name} requires missing reason 'not_missing' when observed"
            )
        return _finite_score(text, name=name, minimum=minimum, maximum=maximum)
    if missing_reason not in MISSING_REASONS - {"not_missing"}:
        raise LearnerEffectStudyError(
            f"missing {name} requires one fixed non-empty missing reason"
        )
    return None


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "learner-effect analysis requires NumPy; install the recognition extra"
        ) from exc
    return np


def _ols_treatment_effect(pre: Any, post: Any, treatment: Any) -> float:
    np = _require_numpy()
    design = np.column_stack((np.ones(len(pre), dtype=np.float64), treatment, pre))
    coefficients, _, rank, _ = np.linalg.lstsq(design, post, rcond=None)
    if int(rank) != 3:
        raise LearnerEffectStudyError(
            "primary ANCOVA design is rank deficient; arm and pre must vary"
        )
    return float(coefficients[1])


def _cluster_bootstrap(
    *,
    pre: Any,
    post: Any,
    treatment: Any,
    cluster_tokens: list[str],
    replicates: int,
    seed: int,
    standardizer: float,
) -> dict[str, Any]:
    np = _require_numpy()
    clusters_by_arm: dict[int, list[str]] = {0: [], 1: []}
    cluster_arm: dict[str, int] = {}
    indices_by_cluster: dict[str, Any] = {}
    for index, (cluster, arm) in enumerate(zip(cluster_tokens, treatment, strict=True)):
        integer_arm = int(arm)
        previous = cluster_arm.setdefault(cluster, integer_arm)
        if previous != integer_arm:
            raise LearnerEffectStudyError("one randomized cluster appears in both arms")
        indices_by_cluster.setdefault(cluster, []).append(index)
    for cluster, arm in cluster_arm.items():
        clusters_by_arm[arm].append(cluster)
    if not clusters_by_arm[0] or not clusters_by_arm[1]:
        raise LearnerEffectStudyError(
            "cluster trial requires at least one cluster per arm"
        )
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_indices: list[int] = []
        for arm in (0, 1):
            arm_clusters = clusters_by_arm[arm]
            sampled = generator.integers(0, len(arm_clusters), size=len(arm_clusters))
            for selected in sampled:
                sampled_indices.extend(indices_by_cluster[arm_clusters[int(selected)]])
        indices = np.asarray(sampled_indices, dtype=np.int64)
        try:
            estimate = (
                _ols_treatment_effect(pre[indices], post[indices], treatment[indices])
                / standardizer
            )
        except LearnerEffectStudyError:
            continue
        if math.isfinite(estimate):
            estimates.append(float(estimate))
    valid_fraction = len(estimates) / replicates
    if len(estimates) < 20 or valid_fraction < 0.9:
        raise LearnerEffectStudyError(
            "fewer than 90% of preregistered cluster bootstrap replicates were valid"
        )
    interval = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return {
        "method": "arm_stratified_cluster_resampling_with_replacement",
        "unit": "randomized_teacher_or_classroom_cluster",
        "replicates_requested": replicates,
        "replicates_valid": len(estimates),
        "valid_fraction": round(valid_fraction, 6),
        "seed": seed,
        "confidence_level": 0.95,
        "percentile_95_ci": [round(float(item), 6) for item in interval],
    }


def analyze_learner_effect_study(
    preregistration_path: str | Path,
    allocation_path: str | Path,
    outcomes_path: str | Path,
    teacher_burden_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze filled real or synthetic tables under the frozen local protocol."""

    np = _require_numpy()
    prereg_path = Path(preregistration_path).expanduser()
    allocation_file = Path(allocation_path).expanduser()
    outcomes_file = Path(outcomes_path).expanduser()
    preregistration, core, core_sha256 = _load_preregistration(prereg_path)
    bindings = preregistration["artifact_bindings"]
    if allocation_file.is_symlink() or not allocation_file.is_file():
        raise LearnerEffectStudyError("allocation CSV is missing or unsafe")
    if outcomes_file.is_symlink() or not outcomes_file.is_file():
        raise LearnerEffectStudyError("outcome CSV is missing or unsafe")
    if _file_sha256(allocation_file) != bindings["allocation_csv_sha256"]:
        raise LearnerEffectStudyError(
            "allocation CSV hash differs from preregistration"
        )
    allocation_rows = _read_csv_exact(allocation_file, ALLOCATION_FIELDS, "allocation")
    outcome_rows = _read_csv_exact(outcomes_file, OUTCOME_FIELDS, "outcome")
    design = core.get("design")
    governance = core.get("governance")
    outcomes = core.get("outcomes")
    plan = core.get("analysis_plan")
    if not all(
        isinstance(value, dict) for value in (design, governance, outcomes, plan)
    ):
        raise LearnerEffectStudyError("preregistration core sections are malformed")
    assert isinstance(design, dict)
    assert isinstance(governance, dict)
    assert isinstance(outcomes, dict)
    assert isinstance(plan, dict)
    if (
        design.get("study_design") != "cluster_randomized_controlled_trial"
        or design.get("cluster_unit") not in {"teacher", "classroom"}
        or outcomes.get("primary_outcome") != "post_score_adjusted_for_pre_score"
        or outcomes.get("primary_model") != "participant_level_ols_post_on_arm_and_pre"
        or outcomes.get("internal_skill_score_as_outcome_prohibited") is not True
        or plan.get("estimand") != "intention_to_treat_by_randomized_cluster_assignment"
        or plan.get("cluster_bootstrap")
        != "arm_stratified_resample_clusters_with_replacement"
        or core.get("protocol_version") != PROTOCOL_VERSION
        or outcomes.get("secondary_outcomes")
        != [
            "transfer_score_adjusted_for_pre_score",
            "retention_score_at_14_to_28_days_adjusted_for_pre_score",
        ]
        or outcomes.get("assessor_must_be_independent_and_blind") is not True
    ):
        raise LearnerEffectStudyError("unsupported or altered preregistered analysis")
    retention_window = outcomes.get("retention_window_days")
    if (
        not isinstance(retention_window, list)
        or len(retention_window) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in retention_window
        )
        or not 14 <= retention_window[0] <= retention_window[1] <= 28
    ):
        raise LearnerEffectStudyError("preregistered retention window is invalid")
    expected_participants = int(design.get("planned_participant_count", -1))
    expected_clusters = int(design.get("planned_cluster_count", -1))
    if len(allocation_rows) != expected_participants:
        raise LearnerEffectStudyError(
            "allocation row count differs from preregistration"
        )
    allocation_by_token: dict[str, dict[str, str]] = {}
    cluster_arms: dict[str, str] = {}
    for row in allocation_rows:
        participant = row["participant_token"].strip()
        cluster = row["cluster_token"].strip()
        arm = row["assigned_arm"].strip()
        if (
            row["preregistration_core_sha256"] != core_sha256
            or not SHA256_RE.fullmatch(participant)
            or not SHA256_RE.fullmatch(cluster)
            or arm not in {"A", "B"}
            or row["randomized"].strip().casefold() != "true"
            or participant in allocation_by_token
        ):
            raise LearnerEffectStudyError(
                "allocation CSV has an invalid or duplicate row"
            )
        prior_arm = cluster_arms.setdefault(cluster, arm)
        if prior_arm != arm:
            raise LearnerEffectStudyError("one cluster is assigned to both arms")
        allocation_by_token[participant] = row
    if len(cluster_arms) != expected_clusters:
        raise LearnerEffectStudyError(
            "observed allocation cluster count differs from plan"
        )
    if (
        abs(
            list(cluster_arms.values()).count("A")
            - list(cluster_arms.values()).count("B")
        )
        > 0
    ):
        raise LearnerEffectStudyError(
            "allocation is not the preregistered balanced cluster trial"
        )

    score_minimum = float(outcomes["score_minimum"])
    score_maximum = float(outcomes["score_maximum"])
    outcome_by_token: dict[str, dict[str, str]] = {}
    pre_values: list[float] = []
    post_itt_values: list[float] = []
    transfer_itt_values: list[float] = []
    retention_itt_values: list[float] = []
    treatment_values: list[int] = []
    cluster_values: list[str] = []
    subgroup_values: list[str] = []
    observed_post_count = 0
    observed_post_by_arm = {"A": 0, "B": 0}
    observed_transfer_count = 0
    observed_retention_count = 0
    grader_blind_count = 0
    adverse_event_yes_count = 0
    protocol_deviation_count = 0
    serious_related_adverse_event_count = 0
    primary_missing_reasons: dict[str, int] = {}
    transfer_missing_reasons: dict[str, int] = {}
    retention_missing_reasons: dict[str, int] = {}
    subgroup_plan = plan.get("subgroup_analysis")
    if not isinstance(subgroup_plan, dict):
        raise LearnerEffectStudyError("preregistered subgroup analysis is malformed")
    subgroup_codes = subgroup_plan.get("predeclared_codes")
    if not isinstance(subgroup_codes, list) or any(
        not isinstance(item, str) or SUBGROUP_CODE_RE.fullmatch(item) is None
        for item in subgroup_codes
    ):
        raise LearnerEffectStudyError("preregistered subgroup codes are invalid")
    for row in outcome_rows:
        participant = row["participant_token"].strip()
        allocation = allocation_by_token.get(participant)
        if (
            allocation is None
            or participant in outcome_by_token
            or row["preregistration_core_sha256"] != core_sha256
            or row["cluster_token"].strip() != allocation["cluster_token"]
        ):
            raise LearnerEffectStudyError(
                "outcome CSV token/cluster mapping differs from allocation"
            )
        pre = _finite_score(
            row["pre_score"].strip(),
            name="pre_score",
            minimum=score_minimum,
            maximum=score_maximum,
        )
        post = _optional_score(
            row["post_score"],
            row["primary_missing_reason"],
            name="post_score",
            minimum=score_minimum,
            maximum=score_maximum,
        )
        transfer = _optional_score(
            row["transfer_score"],
            row["transfer_missing_reason"],
            name="transfer_score",
            minimum=score_minimum,
            maximum=score_maximum,
        )
        retention = _optional_score(
            row["retention_score"],
            row["retention_missing_reason"],
            name="retention_score",
            minimum=score_minimum,
            maximum=score_maximum,
        )
        grader_blind = row["grader_blind"].strip().casefold()
        adverse = row["adverse_event_reported"].strip().casefold()
        severity = row["adverse_event_severity"].strip().casefold()
        relatedness = row["adverse_event_relatedness"].strip().casefold()
        subgroup = row["subgroup_code"].strip()
        if grader_blind not in {"true", "false"}:
            raise LearnerEffectStudyError(
                "grader_blind must be explicitly true or false"
            )
        if adverse not in {"yes", "no"}:
            raise LearnerEffectStudyError(
                "adverse_event_reported must be explicitly yes or no"
            )
        if severity not in ADVERSE_EVENT_SEVERITIES:
            raise LearnerEffectStudyError("adverse_event_severity is invalid")
        if relatedness not in ADVERSE_EVENT_RELATEDNESS:
            raise LearnerEffectStudyError("adverse_event_relatedness is invalid")
        if (adverse == "no" and (severity != "none" or relatedness != "none")) or (
            adverse == "yes" and (severity == "none" or relatedness == "none")
        ):
            raise LearnerEffectStudyError(
                "adverse-event severity and relatedness contradict reported status"
            )
        if subgroup not in subgroup_codes:
            raise LearnerEffectStudyError("subgroup_code was not preregistered")
        retention_day_text = row["retention_assessment_day"].strip()
        if retention is None:
            if retention_day_text:
                raise LearnerEffectStudyError(
                    "missing retention_score cannot declare an assessment day"
                )
        else:
            try:
                retention_day = int(retention_day_text)
            except ValueError as exc:
                raise LearnerEffectStudyError(
                    "observed retention_score requires an integer assessment day"
                ) from exc
            if not retention_window[0] <= retention_day <= retention_window[1]:
                raise LearnerEffectStudyError(
                    "retention assessment is outside the preregistered 2-4 week window"
                )
        protocol_deviation = row["protocol_deviation"].strip()
        if protocol_deviation not in PROTOCOL_DEVIATIONS:
            raise LearnerEffectStudyError(
                "protocol_deviation must use one fixed non-identifying category"
            )
        outcome_by_token[participant] = row
        pre_values.append(pre)
        post_itt_values.append(pre if post is None else post)
        transfer_itt_values.append(pre if transfer is None else transfer)
        retention_itt_values.append(pre if retention is None else retention)
        treatment_values.append(int(allocation["assigned_arm"] == "A"))
        cluster_values.append(allocation["cluster_token"])
        subgroup_values.append(subgroup)
        observed_post_count += int(post is not None)
        observed_post_by_arm[allocation["assigned_arm"]] += int(post is not None)
        observed_transfer_count += int(transfer is not None)
        observed_retention_count += int(retention is not None)
        grader_blind_count += int(grader_blind == "true")
        adverse_event_yes_count += int(adverse == "yes")
        serious_related_adverse_event_count += int(
            adverse == "yes"
            and severity == "serious"
            and relatedness in {"possibly", "probably"}
        )
        protocol_deviation_count += int(protocol_deviation != "none")
        primary_reason = row["primary_missing_reason"].strip()
        transfer_reason = row["transfer_missing_reason"].strip()
        retention_reason = row["retention_missing_reason"].strip()
        primary_missing_reasons[primary_reason] = (
            primary_missing_reasons.get(primary_reason, 0) + 1
        )
        transfer_missing_reasons[transfer_reason] = (
            transfer_missing_reasons.get(transfer_reason, 0) + 1
        )
        retention_missing_reasons[retention_reason] = (
            retention_missing_reasons.get(retention_reason, 0) + 1
        )
    if set(outcome_by_token) != set(allocation_by_token):
        raise LearnerEffectStudyError(
            "outcome CSV must contain every randomized token exactly once for ITT"
        )
    pre_array = np.asarray(pre_values, dtype=np.float64)
    post_array = np.asarray(post_itt_values, dtype=np.float64)
    transfer_array = np.asarray(transfer_itt_values, dtype=np.float64)
    retention_array = np.asarray(retention_itt_values, dtype=np.float64)
    treatment_array = np.asarray(treatment_values, dtype=np.uint8)
    pre_standard_deviation = float(np.std(pre_array, ddof=1))
    if not math.isfinite(pre_standard_deviation) or pre_standard_deviation <= 0:
        raise LearnerEffectStudyError(
            "all-randomized pre-score standardizer must be finite and positive"
        )
    raw_effect = _ols_treatment_effect(pre_array, post_array, treatment_array)
    standardized_effect = raw_effect / pre_standard_deviation
    bootstrap = _cluster_bootstrap(
        pre=pre_array,
        post=post_array,
        treatment=treatment_array,
        cluster_tokens=cluster_values,
        replicates=int(plan["bootstrap_replicates"]),
        seed=int(plan["bootstrap_seed"]),
        standardizer=pre_standard_deviation,
    )
    secondary_seeds = plan.get("secondary_bootstrap_seeds")
    if not isinstance(secondary_seeds, dict) or set(secondary_seeds) != {
        "transfer",
        "retention",
    }:
        raise LearnerEffectStudyError("secondary bootstrap seeds are not frozen")

    def secondary_analysis(values: Any, *, seed_name: str) -> dict[str, Any]:
        raw = _ols_treatment_effect(pre_array, values, treatment_array)
        standardized = raw / pre_standard_deviation
        interval = _cluster_bootstrap(
            pre=pre_array,
            post=values,
            treatment=treatment_array,
            cluster_tokens=cluster_values,
            replicates=int(plan["bootstrap_replicates"]),
            seed=int(secondary_seeds[seed_name]),
            standardizer=pre_standard_deviation,
        )
        return {
            "raw_adjusted_arm_a_minus_b_effect": round(raw, 6),
            "adjusted_standardized_effect": round(standardized, 6),
            "cluster_bootstrap": interval,
            "missing_policy": "baseline_carried_forward_score_equals_pre",
        }

    transfer_analysis = secondary_analysis(transfer_array, seed_name="transfer")
    retention_analysis = secondary_analysis(retention_array, seed_name="retention")
    subgroup_results: list[dict[str, Any]] = []
    released_subgroup_effects: list[float] = []
    minimum_subgroup_cell_size = int(subgroup_plan["minimum_cell_size_per_arm"])
    for subgroup_code in subgroup_codes:
        subgroup_indices = np.asarray(
            [
                index
                for index, code in enumerate(subgroup_values)
                if code == subgroup_code
            ],
            dtype=np.int64,
        )
        arm_counts = {
            "A": int(np.sum(treatment_array[subgroup_indices] == 1)),
            "B": int(np.sum(treatment_array[subgroup_indices] == 0)),
        }
        releasable = min(arm_counts.values()) >= minimum_subgroup_cell_size
        effect: float | None = None
        if releasable:
            subgroup_pre = pre_array[subgroup_indices]
            subgroup_sd = float(np.std(subgroup_pre, ddof=1))
            if math.isfinite(subgroup_sd) and subgroup_sd > 0:
                try:
                    effect = (
                        _ols_treatment_effect(
                            subgroup_pre,
                            post_array[subgroup_indices],
                            treatment_array[subgroup_indices],
                        )
                        / subgroup_sd
                    )
                except LearnerEffectStudyError:
                    effect = None
        if effect is None:
            subgroup_results.append(
                {
                    "subgroup_code": subgroup_code,
                    "small_cell_suppressed": True,
                    "participants_per_arm": None,
                    "adjusted_standardized_effect": None,
                }
            )
        else:
            rounded_effect = round(float(effect), 6)
            released_subgroup_effects.append(rounded_effect)
            subgroup_results.append(
                {
                    "subgroup_code": subgroup_code,
                    "small_cell_suppressed": False,
                    "participants_per_arm": arm_counts,
                    "adjusted_standardized_effect": rounded_effect,
                }
            )
    fairness_gap = (
        round(max(released_subgroup_effects) - min(released_subgroup_effects), 6)
        if len(released_subgroup_effects) >= 2
        else None
    )
    teacher_burden: dict[str, Any] = {
        "status": "external_validation_pending",
        "completed_cluster_count": 0,
        "expected_cluster_count": len(cluster_arms),
        "aggregate_by_arm": None,
        "arm_a_minus_b_total_minutes": None,
        "row_level_cluster_tokens_included": False,
    }
    burden_file_sha256: str | None = None
    if teacher_burden_path is not None:
        burden_file = Path(teacher_burden_path).expanduser()
        burden_rows = _read_csv_exact(
            burden_file, TEACHER_BURDEN_FIELDS, "teacher burden"
        )
        if len(burden_rows) != len(cluster_arms):
            raise LearnerEffectStudyError(
                "teacher burden CSV must contain every randomized cluster exactly once"
            )
        seen_burden_clusters: set[str] = set()
        totals_by_arm: dict[str, list[float]] = {"A": [], "B": []}
        workload_by_arm: dict[str, list[float]] = {"A": [], "B": []}
        missing_count = 0
        for row in burden_rows:
            cluster = row["cluster_token"].strip()
            if (
                row["preregistration_core_sha256"] != core_sha256
                or cluster not in cluster_arms
                or cluster in seen_burden_clusters
            ):
                raise LearnerEffectStudyError(
                    "teacher burden cluster mapping differs from allocation"
                )
            seen_burden_clusters.add(cluster)
            reason = row["burden_missing_reason"].strip()
            value_fields = (
                "preparation_minutes",
                "delivery_minutes",
                "followup_minutes",
                "workload_rating_1_to_5",
            )
            if reason == "not_missing":
                minutes = [
                    _finite_score(
                        row[field].strip(), name=field, minimum=0.0, maximum=10_000.0
                    )
                    for field in value_fields[:3]
                ]
                workload = _finite_score(
                    row["workload_rating_1_to_5"].strip(),
                    name="workload_rating_1_to_5",
                    minimum=1.0,
                    maximum=5.0,
                )
                arm = cluster_arms[cluster]
                totals_by_arm[arm].append(sum(minutes))
                workload_by_arm[arm].append(workload)
            elif reason in BURDEN_MISSING_REASONS - {"not_missing"}:
                if any(row[field].strip() for field in value_fields):
                    raise LearnerEffectStudyError(
                        "missing teacher burden row must leave numeric fields blank"
                    )
                missing_count += 1
            else:
                raise LearnerEffectStudyError(
                    "teacher burden row requires one fixed missing reason"
                )
        burden_file_sha256 = _file_sha256(burden_file)
        aggregates = {
            arm: {
                "completed_cluster_count": len(totals_by_arm[arm]),
                "mean_total_minutes": (
                    round(float(np.mean(totals_by_arm[arm])), 6)
                    if totals_by_arm[arm]
                    else None
                ),
                "mean_workload_rating_1_to_5": (
                    round(float(np.mean(workload_by_arm[arm])), 6)
                    if workload_by_arm[arm]
                    else None
                ),
            }
            for arm in ("A", "B")
        }
        arm_difference = (
            round(
                aggregates["A"]["mean_total_minutes"]
                - aggregates["B"]["mean_total_minutes"],
                6,
            )
            if aggregates["A"]["mean_total_minutes"] is not None
            and aggregates["B"]["mean_total_minutes"] is not None
            else None
        )
        teacher_burden = {
            "status": (
                "complete_for_external_review"
                if missing_count == 0
                else "incomplete_external_validation_pending"
            ),
            "completed_cluster_count": len(cluster_arms) - missing_count,
            "expected_cluster_count": len(cluster_arms),
            "aggregate_by_arm": aggregates,
            "arm_a_minus_b_total_minutes": arm_difference,
            "row_level_cluster_tokens_included": False,
        }
    participant_count = len(allocation_rows)
    primary_coverage = observed_post_count / participant_count
    primary_attrition = 1.0 - primary_coverage
    eligible_by_arm = {
        arm: sum(row["assigned_arm"] == arm for row in allocation_rows)
        for arm in ("A", "B")
    }
    primary_attrition_by_arm = {
        arm: 1.0 - observed_post_by_arm[arm] / eligible_by_arm[arm]
        for arm in ("A", "B")
    }
    differential_attrition = abs(
        primary_attrition_by_arm["A"] - primary_attrition_by_arm["B"]
    )
    transfer_coverage = observed_transfer_count / participant_count
    retention_coverage = observed_retention_count / participant_count
    ci_lower, ci_upper = bootstrap["percentile_95_ci"]
    ethics_id = str(governance.get("ethics_approval_id", "")).strip()
    real_data = (
        governance.get("real_participant_data_declared") is True
        and governance.get("data_origin") == "real"
    )
    gates = {
        "real_participant_data_declared": real_data,
        "ethics_approval_and_consent_declared": bool(
            ethics_id
            and ethics_id.casefold() not in {"pending", "template", "none"}
            and governance.get("informed_consent_or_approved_waiver") is True
        ),
        "preregistration_frozen_before_allocation": governance.get(
            "preregistration_frozen_before_allocation"
        )
        is True,
        "allocation_concealment_procedure_declared": governance.get(
            "allocation_concealment_procedure_declared"
        )
        is True,
        "minimum_preregistered_cluster_count_met": len(cluster_arms)
        >= int(plan["minimum_cluster_count"]),
        "minimum_participant_count_met": participant_count
        >= int(plan["minimum_participant_count"]),
        "primary_coverage_gate_met": primary_coverage
        >= float(plan["minimum_primary_outcome_coverage"]),
        "transfer_coverage_gate_met": transfer_coverage
        >= float(plan["minimum_transfer_outcome_coverage"]),
        "retention_2_to_4_week_coverage_gate_met": retention_coverage
        >= float(plan["minimum_retention_outcome_coverage"]),
        "attrition_within_preregistered_limit": primary_attrition
        <= float(plan["maximum_attrition_fraction"]),
        "differential_attrition_within_preregistered_limit": differential_attrition
        <= float(plan["maximum_differential_attrition_fraction"]),
        "all_randomized_participants_in_itt": len(outcome_by_token)
        == participant_count,
        "all_graders_blind": grader_blind_count == participant_count,
        "adverse_event_reporting_complete": True,
        "no_serious_related_adverse_event": serious_related_adverse_event_count == 0,
        "teacher_burden_reporting_complete": teacher_burden["status"]
        == "complete_for_external_review",
        "subgroup_fairness_estimable_without_small_cell_disclosure": fairness_gap
        is not None,
        "fairness_gap_within_preregistered_limit": fairness_gap is not None
        and fairness_gap
        <= float(subgroup_plan["maximum_allowed_standardized_effect_gap"]),
        "cluster_aware_bootstrap_valid": bootstrap["valid_fraction"] >= 0.9,
        "effect_exceeds_preregistered_minimum": standardized_effect
        > float(plan["minimum_educationally_meaningful_standardized_effect"]),
        "confidence_interval_lower_exceeds_preregistered_minimum": ci_lower
        > float(plan["minimum_educationally_meaningful_standardized_effect"]),
        "internal_skill_score_used_as_learning_outcome": False,
        "intervention_version_and_build_frozen": (
            design.get("intervention_version") != "unfrozen"
            and isinstance(design.get("intervention_build_sha256"), str)
            and SHA256_RE.fullmatch(design["intervention_build_sha256"]) is not None
        ),
        "semantic_lockbox_report_hash_preregistered": isinstance(
            core.get("semantic_lockbox_binding", {}).get("report_sha256"), str
        )
        and SHA256_RE.fullmatch(core["semantic_lockbox_binding"]["report_sha256"])
        is not None,
        "independent_assessor_preregistered": (
            isinstance(governance.get("independent_assessor"), dict)
            and governance["independent_assessor"].get(
                "independent_from_developer_and_intervention_provider_declared"
            )
            is True
            and governance["independent_assessor"].get("organization")
            != "external_validation_pending"
            and governance["independent_assessor"].get("key_id")
            != "external_validation_pending"
        ),
    }
    local_positive_gate = (
        all(
            value
            for key, value in gates.items()
            if key != "internal_skill_score_used_as_learning_outcome"
        )
        and gates["internal_skill_score_used_as_learning_outcome"] is False
    )
    establishment_gates = {
        **{
            key: value
            for key, value in gates.items()
            if key != "internal_skill_score_used_as_learning_outcome"
        },
        "internal_skill_score_excluded_from_learning_outcomes": not gates[
            "internal_skill_score_used_as_learning_outcome"
        ],
        "confirmatory_p_value_from_external_governance_available": False,
        "independent_blind_assessor_signature_verified": False,
        "semantic_lockbox_binding_verified": False,
        "external_evidence_signature_verified": False,
    }
    result = {
        "schema": ANALYSIS_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "study_id": core["study_id"],
        "input_bindings": {
            "preregistration_sha256": _file_sha256(prereg_path),
            "preregistration_core_sha256": core_sha256,
            "allocation_csv_sha256": _file_sha256(allocation_file),
            "completed_outcomes_csv_sha256": _file_sha256(outcomes_file),
            "completed_teacher_burden_csv_sha256": burden_file_sha256,
            "analysis_code_sha256": _file_sha256(Path(__file__).resolve()),
            "semantic_lockbox_report_sha256": core.get(
                "semantic_lockbox_binding", {}
            ).get("report_sha256"),
        },
        "design": {
            "study_design": "cluster_randomized_controlled_trial",
            "cluster_unit": design["cluster_unit"],
            "cluster_count": len(cluster_arms),
            "clusters_per_arm": {
                arm: list(cluster_arms.values()).count(arm) for arm in ("A", "B")
            },
            "randomized_participant_count": participant_count,
            "analysis_is_intention_to_treat": True,
            "assigned_arm_used_regardless_of_protocol_deviation": True,
            "intervention_version": design.get("intervention_version"),
            "intervention_build_sha256": design.get("intervention_build_sha256"),
            "external_assessor_registration": governance.get("independent_assessor"),
        },
        "primary_analysis": {
            "primary_outcome": "post_score_adjusted_for_pre_score",
            "model": "participant_level_ols_post_on_arm_and_pre",
            "missing_post_policy": "baseline_carried_forward_post_equals_pre",
            "raw_adjusted_arm_a_minus_b_effect": round(raw_effect, 6),
            "pre_score_standard_deviation": round(pre_standard_deviation, 6),
            "adjusted_standardized_effect": round(standardized_effect, 6),
            "cluster_bootstrap": bootstrap,
            "minimum_preregistered_effect": float(
                plan["minimum_educationally_meaningful_standardized_effect"]
            ),
            "p_value_computed_by_this_minimum_tool": False,
        },
        "secondary_analyses": {
            "transfer": {
                **transfer_analysis,
                "outcome": "unseen_item_transfer_score_adjusted_for_pre_score",
                "confirmatory": False,
            },
            "retention_2_to_4_weeks": {
                **retention_analysis,
                "outcome": "retention_score_adjusted_for_pre_score",
                "assessment_window_days": retention_window,
                "confirmatory": False,
            },
        },
        "coverage_and_attrition": {
            "eligible_randomized_participant_count": participant_count,
            "observed_primary_outcome_count": observed_post_count,
            "baseline_carried_forward_count": participant_count - observed_post_count,
            "primary_outcome_coverage_fraction": round(primary_coverage, 6),
            "primary_attrition_fraction": round(primary_attrition, 6),
            "primary_attrition_fraction_by_arm": {
                arm: round(value, 6) for arm, value in primary_attrition_by_arm.items()
            },
            "absolute_differential_attrition_fraction": round(
                differential_attrition, 6
            ),
            "maximum_preregistered_differential_attrition_fraction": float(
                plan["maximum_differential_attrition_fraction"]
            ),
            "transfer_outcome_coverage_fraction": round(transfer_coverage, 6),
            "retention_outcome_coverage_fraction": round(retention_coverage, 6),
            "primary_missing_reason_counts": primary_missing_reasons,
            "transfer_missing_reason_counts": transfer_missing_reasons,
            "retention_missing_reason_counts": retention_missing_reasons,
        },
        "subgroup_and_fairness": {
            "analysis_status": (
                "estimable"
                if fairness_gap is not None
                else "external_validation_pending"
            ),
            "subgroup_definitions_registry_sha256": subgroup_plan.get(
                "definitions_registry_sha256"
            ),
            "small_cell_threshold_per_arm": minimum_subgroup_cell_size,
            "subgroups": subgroup_results,
            "max_pairwise_standardized_effect_gap": fairness_gap,
            "preregistered_maximum_gap": float(
                subgroup_plan["maximum_allowed_standardized_effect_gap"]
            ),
            "subgroup_semantics_or_protected_attributes_included": False,
            "causal_subgroup_claim_permitted": False,
        },
        "teacher_burden": teacher_burden,
        "blinding_and_safety": {
            "grader_blind_count": grader_blind_count,
            "adverse_event_yes_count": adverse_event_yes_count,
            "adverse_event_reporting_complete": True,
            "serious_related_adverse_event_count": serious_related_adverse_event_count,
            "automatic_safety_gate_passed": serious_related_adverse_event_count == 0,
            "protocol_deviation_count": protocol_deviation_count,
        },
        "gates": gates,
        "local_preregistered_positive_result_gate": local_positive_gate,
        "eligible_for_external_governance_review": local_positive_gate,
        "learner_effect_establishment_gates": establishment_gates,
        "learner_effect_establishment_gate": all(establishment_gates.values()),
        "learner_effectiveness_established": False,
        "external_validation_status": "external_validation_pending",
        "external_validation_pending_requirements": [
            "independent_blind_assessor_signature",
            "semantic_lockbox_report_signature_and_exact_hash",
            "confirmatory_p_value_and_complete_registered_outcome_reporting",
            "ethics_consent_assessor_independence_and_trusted_key_audit",
        ],
        "claim_boundary": {
            "internal_skill_scores_are_learning_effectiveness": False,
            "row_level_tokens_or_scores_in_result": False,
            "self_declared_ethics_or_data_origin_independently_verified": False,
            "external_governance_signature_verified": False,
            "independent_assessor_signature_verified": False,
            "semantic_lockbox_binding_verified": False,
            "existing_external_evidence_chain_still_required": True,
            "valid_claim": (
                "local preregistered cluster-aware learner-study analysis; not "
                "established learning-effect evidence without external governance"
            ),
        },
        "external_evidence_handoff": {
            "evidence_kind": "real_learner_effectiveness",
            "requires_independent_p_value_and_complete_registered_reporting": True,
            "requires_independent_blind_assessor_attestation": True,
            "requires_exact_semantic_lockbox_report_binding": True,
            "requires_prepare_sign_verify_external_research_evidence_commands": True,
            "analysis_result_file_sha256_must_be_computed_by_external_preparer": True,
        },
        "software": {
            "learner_effect_protocol": PROTOCOL_VERSION,
            "python": sys.version.split()[0],
            "numpy": _version("numpy"),
        },
    }
    return result


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def sign_external_assessment_attestation(
    analysis: dict[str, Any],
    *,
    private_key_pem: bytes,
    assessor_organization: str,
    assessor_key_id: str,
    signed_at_utc: str,
) -> dict[str, Any]:
    """Sign an independent assessor receipt without establishing effectiveness."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise LearnerEffectStudyError("external assessment requires a v2 analysis")
    if analysis.get("learner_effectiveness_established") is not False:
        raise LearnerEffectStudyError("local analysis cannot establish effectiveness")
    if analysis.get("external_validation_status") != "external_validation_pending":
        raise LearnerEffectStudyError(
            "analysis must remain external_validation_pending"
        )
    bindings = analysis.get("input_bindings")
    gates = analysis.get("gates")
    if not isinstance(bindings, dict) or not isinstance(gates, dict):
        raise LearnerEffectStudyError("analysis bindings or gates are malformed")
    for field in (
        "preregistration_core_sha256",
        "completed_outcomes_csv_sha256",
        "completed_teacher_burden_csv_sha256",
        "semantic_lockbox_report_sha256",
        "analysis_code_sha256",
    ):
        if (
            not isinstance(bindings.get(field), str)
            or SHA256_RE.fullmatch(bindings[field]) is None
        ):
            raise LearnerEffectStudyError(
                f"external assessment requires a frozen {field}"
            )
    if gates.get("all_graders_blind") is not True:
        raise LearnerEffectStudyError("external assessment requires blind grading")
    if gates.get("independent_assessor_preregistered") is not True:
        raise LearnerEffectStudyError(
            "independent assessor must be preregistered before allocation"
        )
    if (
        not assessor_organization.strip()
        or assessor_organization == "external_validation_pending"
    ):
        raise LearnerEffectStudyError("assessor organization must be explicit")
    if not STUDY_ID_RE.fullmatch(assessor_key_id):
        raise LearnerEffectStudyError("assessor_key_id is invalid")
    registered_assessor = analysis.get("design", {}).get(
        "external_assessor_registration"
    )
    if (
        not isinstance(registered_assessor, dict)
        or registered_assessor.get("organization") != assessor_organization.strip()
        or registered_assessor.get("key_id") != assessor_key_id
    ):
        raise LearnerEffectStudyError(
            "signing assessor differs from the preregistered independent assessor"
        )
    try:
        signed_at = datetime.fromisoformat(signed_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LearnerEffectStudyError("signed_at_utc is invalid") from exc
    if not signed_at_utc.endswith("Z") or signed_at > datetime.now(timezone.utc):
        raise LearnerEffectStudyError(
            "signed_at_utc must be a non-future UTC timestamp"
        )
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise LearnerEffectStudyError("external assessor key must be Ed25519")
    payload = {
        "schema": EXTERNAL_ASSESSMENT_ATTESTATION_SCHEMA,
        "protocol": EXTERNAL_ASSESSMENT_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "study_id": analysis["study_id"],
        "analysis_sha256": _canonical_sha256(analysis),
        "preregistration_core_sha256": bindings["preregistration_core_sha256"],
        "completed_outcomes_csv_sha256": bindings["completed_outcomes_csv_sha256"],
        "completed_teacher_burden_csv_sha256": bindings[
            "completed_teacher_burden_csv_sha256"
        ],
        "analysis_code_sha256": bindings["analysis_code_sha256"],
        "semantic_lockbox_report_sha256": bindings["semantic_lockbox_report_sha256"],
        "assessor": {
            "organization": assessor_organization.strip(),
            "key_id": assessor_key_id,
            "independent_from_developer_and_intervention_provider_declared": True,
            "blind_to_randomized_arm_during_grading_declared": True,
            "all_registered_outcomes_reported_declared": True,
        },
        "signed_at_utc": signed_at_utc,
        "claim_boundary": {
            "signature_proves_key_control_and_exact_content_only": True,
            "assessor_independence_externally_verified_by_signature_alone": False,
            "semantic_lockbox_is_causal_learning_evidence": False,
            "learner_effectiveness_established": False,
            "external_validation_status": "external_validation_pending",
        },
    }
    signature = private_key.sign(_canonical_bytes(payload))
    return {
        **payload,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "content_sha256": _canonical_sha256(payload),
    }


def verify_external_assessment_attestation(
    attestation: dict[str, Any],
    analysis: dict[str, Any],
    *,
    trusted_public_key_pem: bytes,
    expected_assessor_key_id: str,
    semantic_lockbox_report_path: str | Path,
) -> dict[str, Any]:
    """Verify exact bindings; return a pending, non-effectiveness receipt."""

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(attestation, dict):
        raise LearnerEffectStudyError("attestation must be an object")
    signature_text = attestation.get("signature_base64")
    content_sha256 = attestation.get("content_sha256")
    payload = {
        key: value
        for key, value in attestation.items()
        if key not in {"signature_base64", "content_sha256"}
    }
    if payload.get("schema") != EXTERNAL_ASSESSMENT_ATTESTATION_SCHEMA:
        raise LearnerEffectStudyError(
            "external assessment attestation schema is invalid"
        )
    if payload.get("protocol") != EXTERNAL_ASSESSMENT_PROTOCOL:
        raise LearnerEffectStudyError("external assessment protocol is invalid")
    if content_sha256 != _canonical_sha256(payload):
        raise LearnerEffectStudyError("external assessment content hash mismatch")
    if payload.get("analysis_sha256") != _canonical_sha256(analysis):
        raise LearnerEffectStudyError("attestation is not bound to the exact analysis")
    assessor = payload.get("assessor")
    if (
        not isinstance(assessor, dict)
        or assessor.get("key_id") != expected_assessor_key_id
    ):
        raise LearnerEffectStudyError("external assessor key identity mismatch")
    lockbox_path = Path(semantic_lockbox_report_path).expanduser()
    if lockbox_path.is_symlink() or not lockbox_path.is_file():
        raise LearnerEffectStudyError("semantic lockbox report is missing or unsafe")
    lockbox = read_json(lockbox_path)
    lockbox_payload = (
        {key: value for key, value in lockbox.items() if key != "content_sha256"}
        if isinstance(lockbox, dict)
        else {}
    )
    lockbox_boundary = lockbox_payload.get("claim_boundary", {})
    if (
        not isinstance(lockbox, dict)
        or lockbox.get("schema")
        != "teaching_skill_miner.teacher_agent_semantic_lockbox_report.v1"
        or lockbox.get("passed") is not True
        or lockbox.get("content_sha256") != _canonical_sha256(lockbox_payload)
        or not isinstance(lockbox_boundary, dict)
        or lockbox_boundary.get("behavioral_thresholds_met") is not True
        or lockbox_boundary.get("expert_gold_signature_verified") is not True
        or lockbox_boundary.get(
            "runtime_observations_derived_from_integrity_checked_sessions"
        )
        is not True
        or lockbox_boundary.get("real_students_involved") is not False
        or lockbox_boundary.get("real_learning_effect_established") is not False
    ):
        raise LearnerEffectStudyError(
            "semantic lockbox report is invalid or did not pass"
        )
    if payload.get("semantic_lockbox_report_sha256") != _file_sha256(lockbox_path):
        raise LearnerEffectStudyError(
            "semantic lockbox file hash differs from preregistration"
        )
    public_key = serialization.load_pem_public_key(trusted_public_key_pem)
    if not isinstance(public_key, Ed25519PublicKey):
        raise LearnerEffectStudyError("trusted assessor public key must be Ed25519")
    try:
        signature = base64.b64decode(str(signature_text), validate=True)
        public_key.verify(signature, _canonical_bytes(payload))
    except (ValueError, InvalidSignature) as exc:
        raise LearnerEffectStudyError("external assessor signature is invalid") from exc
    return {
        "signature_verified": True,
        "analysis_sha256": payload["analysis_sha256"],
        "semantic_lockbox_report_sha256": payload["semantic_lockbox_report_sha256"],
        "assessor_key_id": expected_assessor_key_id,
        "learner_effectiveness_established": False,
        "external_validation_status": "external_validation_pending",
        "remaining_external_requirements": [
            "trusted governance must audit assessor independence",
            "confirmatory inference and complete registered reporting remain external",
            "external research evidence signature chain must verify",
        ],
    }


def write_learner_effect_analysis(
    output_path: str | Path,
    result: dict[str, Any],
) -> Path:
    """Write a private aggregate analysis and bind the handoff to exact content."""

    if result.get("schema") != ANALYSIS_SCHEMA:
        raise LearnerEffectStudyError("not a learner-effect analysis result")
    value = json.loads(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return write_json(output_path, value)
