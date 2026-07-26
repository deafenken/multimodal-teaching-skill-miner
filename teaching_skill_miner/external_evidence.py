"""Signed, externally governed evidence for claims that local artifacts cannot prove."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


EVIDENCE_KINDS = (
    "confirmatory_multimodal_gain",
    "real_learner_effectiveness",
)
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExternalEvidenceError(ValueError):
    """Raised when external research evidence is malformed or untrustworthy."""


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
        raise ExternalEvidenceError(
            "external evidence is not canonical finite JSON"
        ) from exc
    return payload.encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(name: str, value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExternalEvidenceError(f"{name} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ExternalEvidenceError(f"{name} is invalid") from exc
    if parsed > datetime.now(timezone.utc):
        raise ExternalEvidenceError(f"{name} is in the future")
    return parsed


def _require_exact_keys(
    name: str,
    value: Any,
    expected: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalEvidenceError(f"{name} must be one JSON object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise ExternalEvidenceError(
            f"{name} fields do not match the protocol; missing={missing}, extra={extra}"
        )
    return dict(value)


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalEvidenceError(f"{name} must be a non-empty string")
    return value.strip()


def _require_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ExternalEvidenceError(
            f"{name} must contain 6-128 safe identifier characters"
        )
    return value


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ExternalEvidenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ExternalEvidenceError(f"{name} must be boolean")
    return value


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExternalEvidenceError(f"{name} must be an integer >= {minimum}")
    return value


def _require_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExternalEvidenceError(f"{name} must be numeric")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ExternalEvidenceError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ExternalEvidenceError(f"{name} must be finite")
    if minimum is not None and (
        numeric < minimum or (minimum_exclusive and numeric == minimum)
    ):
        relation = ">" if minimum_exclusive else ">="
        raise ExternalEvidenceError(f"{name} must be {relation} {minimum}")
    if maximum is not None and (
        numeric > maximum or (maximum_exclusive and numeric == maximum)
    ):
        relation = "<" if maximum_exclusive else "<="
        raise ExternalEvidenceError(f"{name} must be {relation} {maximum}")
    return numeric


def _require_string_list(name: str, value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ExternalEvidenceError(
            f"{name} must be a non-empty list of non-empty strings"
        )
    normalized = [item.strip() for item in value]
    if len(set(normalized)) != len(normalized):
        raise ExternalEvidenceError(f"{name} must not contain duplicates")
    return normalized


def _common_sections(evidence: dict[str, Any]) -> dict[str, Any]:
    preregistration = _require_exact_keys(
        "preregistration",
        evidence["preregistration"],
        {
            "registry_name",
            "registration_id",
            "protocol_sha256",
            "statistical_analysis_plan_sha256",
            "frozen_before_outcome_access",
        },
    )
    _require_nonempty_string(
        "preregistration.registry_name", preregistration["registry_name"]
    )
    _require_nonempty_string(
        "preregistration.registration_id", preregistration["registration_id"]
    )
    _require_sha256(
        "preregistration.protocol_sha256", preregistration["protocol_sha256"]
    )
    _require_sha256(
        "preregistration.statistical_analysis_plan_sha256",
        preregistration["statistical_analysis_plan_sha256"],
    )
    if preregistration["frozen_before_outcome_access"] is not True:
        raise ExternalEvidenceError(
            "preregistration.frozen_before_outcome_access must be true"
        )

    governance = _require_exact_keys(
        "governance",
        evidence["governance"],
        {
            "independent_external_governance",
            "data_or_analysis_independent_of_system_developers",
            "data_not_used_to_modify_system_or_claim_thresholds",
            "conflicts_of_interest_disclosed",
            "complete_registered_outcome_reporting",
            "trusted_key_provisioned_out_of_band",
        },
    )
    for field, value in governance.items():
        if value is not True:
            raise ExternalEvidenceError(f"governance.{field} must be true")

    population = _require_exact_keys(
        "population",
        evidence["population"],
        {
            "real_world_data",
            "data_context",
            "site_count",
            "independent_site_count",
            "participant_count",
            "eligible_unit_count",
            "evaluated_unit_count",
            "excluded_unit_count",
            "exclusion_reason_counts",
            "coverage_fraction",
            "coverage_evidence_sha256",
            "identity_disjoint_from_development",
        },
    )
    if population["real_world_data"] is not True:
        raise ExternalEvidenceError("population.real_world_data must be true")
    data_context = _require_nonempty_string(
        "population.data_context", population["data_context"]
    )
    if data_context not in ("real_classroom", "real_learner_study"):
        raise ExternalEvidenceError("population.data_context is unsupported")
    if population["identity_disjoint_from_development"] is not True:
        raise ExternalEvidenceError(
            "population.identity_disjoint_from_development must be true"
        )
    site_count = _require_int("population.site_count", population["site_count"], minimum=1)
    independent_site_count = _require_int(
        "population.independent_site_count",
        population["independent_site_count"],
        minimum=1,
    )
    if independent_site_count > site_count:
        raise ExternalEvidenceError(
            "population.independent_site_count cannot exceed site_count"
        )
    participant_count = _require_int(
        "population.participant_count", population["participant_count"], minimum=1
    )
    eligible = _require_int(
        "population.eligible_unit_count",
        population["eligible_unit_count"],
        minimum=1,
    )
    evaluated = _require_int(
        "population.evaluated_unit_count",
        population["evaluated_unit_count"],
        minimum=1,
    )
    excluded = _require_int(
        "population.excluded_unit_count", population["excluded_unit_count"]
    )
    reasons = population["exclusion_reason_counts"]
    if not isinstance(reasons, Mapping):
        raise ExternalEvidenceError(
            "population.exclusion_reason_counts must be one JSON object"
        )
    reason_total = 0
    for reason, count in reasons.items():
        _require_nonempty_string("population exclusion reason", reason)
        reason_total += _require_int(
            f"population.exclusion_reason_counts.{reason}", count, minimum=1
        )
    if eligible != evaluated + excluded:
        raise ExternalEvidenceError(
            "population eligible count must equal evaluated plus excluded counts"
        )
    if reason_total != excluded:
        raise ExternalEvidenceError(
            "population exclusion reason counts must sum to excluded_unit_count"
        )
    coverage = _require_number(
        "population.coverage_fraction",
        population["coverage_fraction"],
        minimum=0.0,
        maximum=1.0,
    )
    expected_coverage = evaluated / eligible
    if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
        raise ExternalEvidenceError(
            "population.coverage_fraction does not match evaluated/eligible counts"
        )
    _require_sha256(
        "population.coverage_evidence_sha256",
        population["coverage_evidence_sha256"],
    )

    bindings = _require_exact_keys(
        "artifact_bindings",
        evidence["artifact_bindings"],
        {
            "dataset_or_study_data_fingerprint",
            "evaluated_system_fingerprint",
            "comparator_fingerprint",
            "analysis_code_sha256",
            "statistical_output_sha256",
            "aggregate_result_table_sha256",
        },
    )
    for field, value in bindings.items():
        _require_sha256(f"artifact_bindings.{field}", value)
    if (
        bindings["evaluated_system_fingerprint"]
        == bindings["comparator_fingerprint"]
    ):
        raise ExternalEvidenceError(
            "evaluated system and comparator fingerprints must differ"
        )
    return {
        "population": population,
        "participant_count": participant_count,
        "eligible": eligible,
        "evaluated": evaluated,
        "coverage": coverage,
    }


def _validate_multimodal_claim(
    claim: Any,
    common: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    value = _require_exact_keys(
        "claim",
        claim,
        {
            "claim_type",
            "evaluation_design",
            "baseline_modalities",
            "added_modalities",
            "primary_metric",
            "baseline_estimate",
            "multimodal_estimate",
            "absolute_gain",
            "confidence_level",
            "confidence_interval_lower",
            "confidence_interval_upper",
            "p_value",
            "alpha",
            "minimum_confirmatory_gain",
            "paired_cluster_aware_inference",
            "claim_cluster_field",
            "claim_cluster_count",
            "all_registered_primary_analyses_reported",
            "multimodal_gain_established",
        },
    )
    if value["claim_type"] != "confirmatory_multimodal_gain":
        raise ExternalEvidenceError("claim.claim_type does not match evidence_kind")
    if value["evaluation_design"] != "paired_same_samples_external_lockbox":
        raise ExternalEvidenceError(
            "confirmatory multimodal evidence requires a paired external lockbox design"
        )
    baseline_modalities = _require_string_list(
        "claim.baseline_modalities", value["baseline_modalities"]
    )
    added_modalities = _require_string_list(
        "claim.added_modalities", value["added_modalities"]
    )
    if set(baseline_modalities) & set(added_modalities):
        raise ExternalEvidenceError(
            "baseline_modalities and added_modalities must be disjoint"
        )
    if value["primary_metric"] not in (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
    ):
        raise ExternalEvidenceError("claim.primary_metric is unsupported")
    baseline = _require_number(
        "claim.baseline_estimate", value["baseline_estimate"], minimum=0.0, maximum=1.0
    )
    multimodal = _require_number(
        "claim.multimodal_estimate",
        value["multimodal_estimate"],
        minimum=0.0,
        maximum=1.0,
    )
    gain = _require_number(
        "claim.absolute_gain", value["absolute_gain"], minimum=-1.0, maximum=1.0
    )
    gain_recomputes = math.isclose(
        gain, multimodal - baseline, rel_tol=0.0, abs_tol=1e-12
    )
    if not gain_recomputes:
        raise ExternalEvidenceError(
            "claim.absolute_gain does not equal multimodal minus baseline estimate"
        )
    confidence_level = _require_number(
        "claim.confidence_level",
        value["confidence_level"],
        minimum=0.90,
        maximum=1.0,
        maximum_exclusive=True,
    )
    ci_lower = _require_number(
        "claim.confidence_interval_lower",
        value["confidence_interval_lower"],
        minimum=-1.0,
        maximum=1.0,
    )
    ci_upper = _require_number(
        "claim.confidence_interval_upper",
        value["confidence_interval_upper"],
        minimum=-1.0,
        maximum=1.0,
    )
    if ci_lower > gain or gain > ci_upper:
        raise ExternalEvidenceError(
            "claim confidence interval must contain the reported gain"
        )
    p_value = _require_number(
        "claim.p_value", value["p_value"], minimum=0.0, maximum=1.0
    )
    alpha = _require_number(
        "claim.alpha",
        value["alpha"],
        minimum=0.0,
        maximum=0.05,
        minimum_exclusive=True,
    )
    minimum_gain = _require_number(
        "claim.minimum_confirmatory_gain",
        value["minimum_confirmatory_gain"],
        minimum=0.0,
        maximum=1.0,
        maximum_exclusive=True,
    )
    cluster_count = _require_int(
        "claim.claim_cluster_count", value["claim_cluster_count"], minimum=1
    )
    cluster_field = _require_nonempty_string(
        "claim.claim_cluster_field", value["claim_cluster_field"]
    )
    if cluster_field not in (
        "session_id",
        "participant_id",
        "teacher_id",
        "classroom_id",
        "site_id",
    ):
        raise ExternalEvidenceError(
            "claim.claim_cluster_field is not a recognized independent cluster unit"
        )
    _require_bool(
        "claim.paired_cluster_aware_inference",
        value["paired_cluster_aware_inference"],
    )
    _require_bool(
        "claim.all_registered_primary_analyses_reported",
        value["all_registered_primary_analyses_reported"],
    )
    declared = _require_bool(
        "claim.multimodal_gain_established",
        value["multimodal_gain_established"],
    )
    gates = {
        "coverage_at_least_0_8": common["coverage"] >= 0.8,
        "at_least_ten_participants": common["participant_count"] >= 10,
        "at_least_ten_independent_claim_clusters": cluster_count >= 10,
        "cluster_count_within_evaluated_units": cluster_count <= common["evaluated"],
        "paired_cluster_aware_inference": value[
            "paired_cluster_aware_inference"
        ]
        is True,
        "all_registered_primary_analyses_reported": value[
            "all_registered_primary_analyses_reported"
        ]
        is True,
        "registered_minimum_gain_is_positive": minimum_gain > 0.0,
        "positive_gain_exceeds_registered_minimum": gain > minimum_gain,
        "confidence_interval_exceeds_registered_minimum": ci_lower > minimum_gain,
        "confidence_level_consistent_with_alpha": confidence_level >= 1.0 - alpha,
        "confirmatory_p_value_passes": p_value <= alpha,
    }
    established = all(gates.values())
    if declared != established:
        raise ExternalEvidenceError(
            "claim.multimodal_gain_established does not equal the recomputed gates"
        )
    return gates, {
        "primary_metric": value["primary_metric"],
        "baseline_estimate": baseline,
        "multimodal_estimate": multimodal,
        "absolute_gain": gain,
        "confidence_level": confidence_level,
        "confidence_interval": [ci_lower, ci_upper],
        "p_value": p_value,
        "alpha": alpha,
        "minimum_confirmatory_gain": minimum_gain,
        "claim_cluster_count": cluster_count,
    }


def _validate_learner_claim(
    claim: Any,
    common: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    value = _require_exact_keys(
        "claim",
        claim,
        {
            "claim_type",
            "study_design",
            "randomization_unit",
            "comparator_description",
            "primary_outcome_name",
            "effect_measure",
            "effect_direction",
            "adjusted_effect_estimate",
            "confidence_level",
            "confidence_interval_lower",
            "confidence_interval_upper",
            "p_value",
            "alpha",
            "minimum_educationally_meaningful_effect",
            "random_assignment",
            "allocation_concealment",
            "intention_to_treat",
            "independent_outcome_assessment",
            "randomized_unit_count",
            "analysis_cluster_count",
            "maximum_preregistered_attrition_fraction",
            "observed_attrition_fraction",
            "attrition_handled_under_preregistered_plan",
            "adverse_events_reported",
            "all_registered_primary_outcomes_reported",
            "ethics_approval_id",
            "informed_consent_or_approved_waiver",
            "learner_effectiveness_established",
        },
    )
    if value["claim_type"] != "real_learner_effectiveness":
        raise ExternalEvidenceError("claim.claim_type does not match evidence_kind")
    designs = {
        "individual_randomized_controlled_trial": {"learner"},
        "cluster_randomized_controlled_trial": {"classroom", "teacher"},
    }
    design = _require_nonempty_string("claim.study_design", value["study_design"])
    if design not in designs or value["randomization_unit"] not in designs.get(
        design, set()
    ):
        raise ExternalEvidenceError(
            "study_design and randomization_unit are inconsistent"
        )
    _require_nonempty_string(
        "claim.comparator_description", value["comparator_description"]
    )
    _require_nonempty_string(
        "claim.primary_outcome_name", value["primary_outcome_name"]
    )
    if value["effect_measure"] != "standardized_mean_difference":
        raise ExternalEvidenceError(
            "learner evidence currently requires standardized_mean_difference"
        )
    if value["effect_direction"] != "positive_favors_teaching_skill":
        raise ExternalEvidenceError(
            "learner effect direction must be positive_favors_teaching_skill"
        )
    effect = _require_number(
        "claim.adjusted_effect_estimate",
        value["adjusted_effect_estimate"],
        minimum=-10.0,
        maximum=10.0,
    )
    confidence_level = _require_number(
        "claim.confidence_level",
        value["confidence_level"],
        minimum=0.90,
        maximum=1.0,
        maximum_exclusive=True,
    )
    ci_lower = _require_number(
        "claim.confidence_interval_lower",
        value["confidence_interval_lower"],
        minimum=-10.0,
        maximum=10.0,
    )
    ci_upper = _require_number(
        "claim.confidence_interval_upper",
        value["confidence_interval_upper"],
        minimum=-10.0,
        maximum=10.0,
    )
    if ci_lower > effect or effect > ci_upper:
        raise ExternalEvidenceError(
            "claim confidence interval must contain the adjusted learner effect"
        )
    p_value = _require_number(
        "claim.p_value", value["p_value"], minimum=0.0, maximum=1.0
    )
    alpha = _require_number(
        "claim.alpha",
        value["alpha"],
        minimum=0.0,
        maximum=0.05,
        minimum_exclusive=True,
    )
    minimum_effect = _require_number(
        "claim.minimum_educationally_meaningful_effect",
        value["minimum_educationally_meaningful_effect"],
        minimum=0.0,
        maximum=10.0,
    )
    randomized_units = _require_int(
        "claim.randomized_unit_count", value["randomized_unit_count"], minimum=1
    )
    analysis_clusters = _require_int(
        "claim.analysis_cluster_count", value["analysis_cluster_count"]
    )
    max_attrition = _require_number(
        "claim.maximum_preregistered_attrition_fraction",
        value["maximum_preregistered_attrition_fraction"],
        minimum=0.0,
        maximum=0.2,
    )
    attrition = _require_number(
        "claim.observed_attrition_fraction",
        value["observed_attrition_fraction"],
        minimum=0.0,
        maximum=1.0,
    )
    count_attrition = 1.0 - common["coverage"]
    if not math.isclose(attrition, count_attrition, rel_tol=0.0, abs_tol=1e-12):
        raise ExternalEvidenceError(
            "claim.observed_attrition_fraction does not match population counts"
        )
    for field in (
        "random_assignment",
        "allocation_concealment",
        "intention_to_treat",
        "independent_outcome_assessment",
        "attrition_handled_under_preregistered_plan",
        "adverse_events_reported",
        "all_registered_primary_outcomes_reported",
        "informed_consent_or_approved_waiver",
    ):
        _require_bool(f"claim.{field}", value[field])
    _require_nonempty_string("claim.ethics_approval_id", value["ethics_approval_id"])
    declared = _require_bool(
        "claim.learner_effectiveness_established",
        value["learner_effectiveness_established"],
    )
    individual_design_ok = bool(
        design == "individual_randomized_controlled_trial"
        and randomized_units >= 30
        and randomized_units == common["participant_count"]
        and analysis_clusters == 0
    )
    cluster_design_ok = bool(
        design == "cluster_randomized_controlled_trial"
        and randomized_units >= 6
        and analysis_clusters == randomized_units
    )
    gates = {
        "at_least_thirty_participants": common["participant_count"] >= 30,
        "coverage_at_least_0_8": common["coverage"] >= 0.8,
        "valid_randomized_design_units": individual_design_ok or cluster_design_ok,
        "random_assignment": value["random_assignment"] is True,
        "allocation_concealment": value["allocation_concealment"] is True,
        "intention_to_treat": value["intention_to_treat"] is True,
        "independent_outcome_assessment": value[
            "independent_outcome_assessment"
        ]
        is True,
        "attrition_within_preregistered_limit": attrition <= max_attrition,
        "attrition_handled_under_preregistered_plan": value[
            "attrition_handled_under_preregistered_plan"
        ]
        is True,
        "ethics_and_consent_documented": bool(
            value["ethics_approval_id"].strip()
            and value["informed_consent_or_approved_waiver"] is True
        ),
        "complete_outcomes_and_harms_reported": bool(
            value["adverse_events_reported"] is True
            and value["all_registered_primary_outcomes_reported"] is True
        ),
        "registered_meaningful_effect_is_positive": minimum_effect > 0.0,
        "effect_exceeds_registered_minimum": effect > minimum_effect,
        "confidence_interval_exceeds_registered_minimum": ci_lower
        > minimum_effect,
        "confidence_level_consistent_with_alpha": confidence_level >= 1.0 - alpha,
        "confirmatory_p_value_passes": p_value <= alpha,
    }
    established = all(gates.values())
    if declared != established:
        raise ExternalEvidenceError(
            "claim.learner_effectiveness_established does not equal the recomputed gates"
        )
    return gates, {
        "study_design": design,
        "randomization_unit": value["randomization_unit"],
        "primary_outcome_name": value["primary_outcome_name"],
        "effect_measure": value["effect_measure"],
        "adjusted_effect_estimate": effect,
        "confidence_level": confidence_level,
        "confidence_interval": [ci_lower, ci_upper],
        "p_value": p_value,
        "alpha": alpha,
        "minimum_educationally_meaningful_effect": minimum_effect,
        "participant_count": common["participant_count"],
        "randomized_unit_count": randomized_units,
        "analysis_cluster_count": analysis_clusters,
        "observed_attrition_fraction": attrition,
    }


def external_research_evidence_fingerprint(evidence: Mapping[str, Any]) -> str:
    """Fingerprint every evidence field except the fingerprint itself."""

    if not isinstance(evidence, Mapping):
        raise ExternalEvidenceError("external evidence must be one JSON object")
    payload = dict(evidence)
    payload.pop("evidence_fingerprint", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_external_research_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _require_exact_keys(
        "external evidence",
        evidence,
        {
            "schema_version",
            "protocol",
            "evidence_id",
            "evidence_kind",
            "study_id",
            "registered_at_utc",
            "completed_at_utc",
            "preregistration",
            "governance",
            "population",
            "artifact_bindings",
            "claim",
            "evidence_fingerprint",
        },
    )
    if (
        value["schema_version"] != "1.0"
        or value["protocol"] != "externally_governed_research_evidence_v1"
    ):
        raise ExternalEvidenceError("unsupported external research evidence protocol")
    evidence_id = _require_id("evidence_id", value["evidence_id"])
    _require_id("study_id", value["study_id"])
    evidence_kind = value["evidence_kind"]
    if evidence_kind not in EVIDENCE_KINDS:
        raise ExternalEvidenceError("unsupported external evidence_kind")
    if expected_kind is not None and evidence_kind != expected_kind:
        raise ExternalEvidenceError(
            f"expected evidence_kind {expected_kind}, received {evidence_kind}"
        )
    registered = _parse_utc("registered_at_utc", value["registered_at_utc"])
    completed = _parse_utc("completed_at_utc", value["completed_at_utc"])
    if completed < registered:
        raise ExternalEvidenceError(
            "completed_at_utc cannot precede registered_at_utc"
        )
    common = _common_sections(value)
    if evidence_kind == "confirmatory_multimodal_gain":
        if value["population"]["data_context"] != "real_classroom":
            raise ExternalEvidenceError(
                "confirmatory multimodal gain requires real_classroom data_context"
            )
        gates, metrics = _validate_multimodal_claim(value["claim"], common)
    else:
        if value["population"]["data_context"] != "real_learner_study":
            raise ExternalEvidenceError(
                "learner effectiveness requires real_learner_study data_context"
            )
        if common["eligible"] != common["participant_count"]:
            raise ExternalEvidenceError(
                "learner evidence eligible_unit_count must equal participant_count"
            )
        gates, metrics = _validate_learner_claim(value["claim"], common)
    expected_fingerprint = external_research_evidence_fingerprint(value)
    if value["evidence_fingerprint"] != expected_fingerprint:
        raise ExternalEvidenceError("external evidence fingerprint mismatch")
    established = all(gates.values())
    return value, {
        "valid": True,
        "evidence_id": evidence_id,
        "evidence_kind": evidence_kind,
        "evidence_fingerprint": expected_fingerprint,
        "claim_established": established,
        "gates": gates,
        "metrics": metrics,
        "trust_boundary": (
            "Content integrity is cryptographically verifiable only after signature "
            "verification; signer independence must come from an out-of-band trusted key."
        ),
    }


def finalize_external_research_evidence(
    draft: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the canonical fingerprint after validating a complete evidence draft."""

    if not isinstance(draft, Mapping):
        raise ExternalEvidenceError("external evidence draft must be one JSON object")
    value = dict(draft)
    observed = value.pop("evidence_fingerprint", None)
    fingerprint = external_research_evidence_fingerprint(value)
    if observed is not None and observed != fingerprint:
        raise ExternalEvidenceError("existing external evidence fingerprint is stale")
    value["evidence_fingerprint"] = fingerprint
    _validate_external_research_evidence(value)
    return value


def validate_external_research_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Validate semantics and return independently recomputed claim gates."""

    _, report = _validate_external_research_evidence(
        evidence, expected_kind=expected_kind
    )
    return report


def sign_external_research_evidence(
    evidence: Mapping[str, Any],
    *,
    private_key_pem: bytes,
    issuer: str,
    key_id: str,
    signer_role: str = "independent_external_research_governance",
) -> dict[str, Any]:
    """Sign validated aggregate evidence with an externally governed Ed25519 key."""

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 external evidence requires `pip install -e '.[recognition]'`"
        ) from exc
    validated, _ = _validate_external_research_evidence(evidence)
    issuer_value = _require_nonempty_string("issuer", issuer)
    key_id_value = _require_id("key_id", key_id)
    if signer_role != "independent_external_research_governance":
        raise ExternalEvidenceError(
            "signer_role must be independent_external_research_governance"
        )
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ExternalEvidenceError("private key must be Ed25519")
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    statement = {
        "schema_version": "1.0",
        "protocol": "external_research_evidence_signature_statement_v1",
        "issuer": issuer_value,
        "key_id": key_id_value,
        "signer_role": signer_role,
        "signed_at_utc": _utc_now(),
        "evidence": validated,
    }
    signature = private_key.sign(_canonical_bytes(statement))
    return {
        "schema_version": "1.0",
        "protocol": "ed25519_external_research_evidence_attestation_v1",
        "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
        "statement": statement,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_external_research_evidence_attestation(
    attestation: Mapping[str, Any],
    *,
    trusted_public_key_pem: bytes,
    expected_kind: str | None = None,
    expected_evidence: Mapping[str, Any] | None = None,
    expected_system_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Verify signature, trusted-key binding, exact manifest, and semantic gates."""

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ed25519 external evidence requires `pip install -e '.[recognition]'`"
        ) from exc
    outer = _require_exact_keys(
        "external evidence attestation",
        attestation,
        {
            "schema_version",
            "protocol",
            "public_key_sha256",
            "statement",
            "signature_base64",
        },
    )
    if (
        outer["schema_version"] != "1.0"
        or outer["protocol"]
        != "ed25519_external_research_evidence_attestation_v1"
    ):
        raise ExternalEvidenceError(
            "unsupported external research evidence attestation protocol"
        )
    statement = _require_exact_keys(
        "external evidence signature statement",
        outer["statement"],
        {
            "schema_version",
            "protocol",
            "issuer",
            "key_id",
            "signer_role",
            "signed_at_utc",
            "evidence",
        },
    )
    if (
        statement["schema_version"] != "1.0"
        or statement["protocol"]
        != "external_research_evidence_signature_statement_v1"
    ):
        raise ExternalEvidenceError(
            "unsupported external research evidence signature statement"
        )
    issuer = _require_nonempty_string("statement.issuer", statement["issuer"])
    key_id = _require_id("statement.key_id", statement["key_id"])
    if statement["signer_role"] != "independent_external_research_governance":
        raise ExternalEvidenceError("external evidence signer role is not acceptable")
    signed_at = _parse_utc("statement.signed_at_utc", statement["signed_at_utc"])
    evidence_raw = statement["evidence"]
    if not isinstance(evidence_raw, Mapping):
        raise ExternalEvidenceError("signed external evidence manifest is missing")
    validated, report = _validate_external_research_evidence(
        evidence_raw, expected_kind=expected_kind
    )
    completed_at = _parse_utc("completed_at_utc", validated["completed_at_utc"])
    if signed_at < completed_at:
        raise ExternalEvidenceError(
            "external evidence was signed before the study completion timestamp"
        )
    if expected_evidence is not None and dict(expected_evidence) != validated:
        raise ExternalEvidenceError(
            "standalone external evidence manifest differs from signed evidence"
        )
    system_binding_verified = False
    if expected_system_fingerprint is not None:
        expected_system_fingerprint = _require_sha256(
            "expected_system_fingerprint", expected_system_fingerprint
        )
        observed_system_fingerprint = validated["artifact_bindings"][
            "evaluated_system_fingerprint"
        ]
        if observed_system_fingerprint != expected_system_fingerprint:
            raise ExternalEvidenceError(
                "external evidence evaluated-system fingerprint does not match "
                "the locally supplied system artifact"
            )
        system_binding_verified = True
    public_key = serialization.load_pem_public_key(trusted_public_key_pem)
    if not isinstance(public_key, Ed25519PublicKey):
        raise ExternalEvidenceError("trusted public key must be Ed25519")
    public_der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_fingerprint = hashlib.sha256(public_der).hexdigest()
    _require_sha256("attestation.public_key_sha256", outer["public_key_sha256"])
    if outer["public_key_sha256"] != public_fingerprint:
        raise ExternalEvidenceError(
            "external evidence attestation public-key fingerprint mismatch"
        )
    try:
        signature = base64.b64decode(
            str(outer["signature_base64"]), validate=True
        )
        public_key.verify(signature, _canonical_bytes(statement))
    except (ValueError, InvalidSignature) as exc:
        raise ExternalEvidenceError(
            "external research evidence signature verification failed"
        ) from exc
    return {
        **report,
        "signature_verified": True,
        "issuer": issuer,
        "key_id": key_id,
        "signer_role": statement["signer_role"],
        "signed_at_utc": statement["signed_at_utc"],
        "public_key_sha256": public_fingerprint,
        "attestation_fingerprint": hashlib.sha256(
            _canonical_bytes(outer)
        ).hexdigest(),
        "trusted_key_provisioned_out_of_band": True,
        "signature_alone_proves_signer_independence": False,
        "system_artifact_binding_verified": system_binding_verified,
        "evaluated_system_fingerprint": validated["artifact_bindings"][
            "evaluated_system_fingerprint"
        ],
    }


def verify_external_research_evidence_files(
    manifest_path: str | Path,
    attestation_path: str | Path,
    trusted_public_key_path: str | Path,
    *,
    expected_kind: str | None = None,
    expected_system_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read and verify the three-file external evidence transport contract."""

    from .io_utils import read_json

    manifest_file = Path(manifest_path).resolve()
    attestation_file = Path(attestation_path).resolve()
    public_key_file = Path(trusted_public_key_path).resolve()
    for name, path in (
        ("manifest", manifest_file),
        ("attestation", attestation_file),
        ("trusted public key", public_key_file),
    ):
        if not path.is_file():
            raise ExternalEvidenceError(f"external evidence {name} file does not exist")
    system_artifact_file = (
        Path(expected_system_artifact_path).resolve()
        if expected_system_artifact_path is not None
        else None
    )
    expected_system_fingerprint = None
    if system_artifact_file is not None:
        if not system_artifact_file.is_file():
            raise ExternalEvidenceError(
                "expected evaluated system artifact file does not exist"
            )
        expected_system_fingerprint = _sha256_file(system_artifact_file)
    manifest = read_json(manifest_file)
    attestation = read_json(attestation_file)
    if not isinstance(manifest, Mapping) or not isinstance(attestation, Mapping):
        raise ExternalEvidenceError(
            "external evidence manifest and attestation must be JSON objects"
        )
    report = verify_external_research_evidence_attestation(
        attestation,
        trusted_public_key_pem=public_key_file.read_bytes(),
        expected_kind=expected_kind,
        expected_evidence=manifest,
        expected_system_fingerprint=expected_system_fingerprint,
    )
    return {
        **report,
        "manifest_path": str(manifest_file),
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "attestation_path": str(attestation_file),
        "attestation_sha256": hashlib.sha256(
            attestation_file.read_bytes()
        ).hexdigest(),
        "trusted_public_key_path": str(public_key_file),
        "expected_system_artifact_path": (
            str(system_artifact_file) if system_artifact_file is not None else None
        ),
        "expected_system_artifact_sha256": expected_system_fingerprint,
    }
