"""Privacy-minimizing cluster-randomized learner-effect study tooling.

The module creates a preregistration package and analyzes genuinely completed
tables.  It never creates participant identities or positive learning-effect
evidence.  Local analysis remains explicitly unestablished until the existing
externally governed evidence and signature chain validates the real study.
"""

from __future__ import annotations

import csv
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


PREREGISTRATION_SCHEMA = "teaching_skill_miner.learner_effect_preregistration.v1"
ANALYSIS_SCHEMA = "teaching_skill_miner.learner_effect_analysis.v1"
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
    "retention_score",
    "primary_missing_reason",
    "retention_missing_reason",
    "grader_blind",
    "adverse_event_reported",
    "protocol_deviation",
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


class LearnerEffectStudyError(ValueError):
    """Raised when a study package or filled table fails closed validation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
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
        raise LearnerEffectStudyError("study artifact is not canonical finite JSON") from exc


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
    minimum_meaningful_effect: float,
    score_minimum: float,
    score_maximum: float,
    bootstrap_replicates: int,
    data_origin: str,
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
    if 1.0 - float(minimum_primary_coverage) > float(maximum_attrition_fraction) + 1e-12:
        raise LearnerEffectStudyError(
            "coverage and attrition gates are internally inconsistent"
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
        raise LearnerEffectStudyError("data_origin must be template, synthetic, or real")


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
        minimum_meaningful_effect=minimum_meaningful_effect,
        score_minimum=score_minimum,
        score_maximum=score_maximum,
        bootstrap_replicates=bootstrap_replicates,
        data_origin=data_origin,
    )
    if (
        not isinstance(randomization_seed, int)
        or isinstance(randomization_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or isinstance(bootstrap_seed, bool)
    ):
        raise LearnerEffectStudyError("randomization and bootstrap seeds must be integers")
    if not intervention_description.strip() or not comparator_description.strip():
        raise LearnerEffectStudyError("both arm descriptions must be non-empty")
    if not isinstance(informed_consent_or_approved_waiver, bool) or not isinstance(
        preregistration_frozen_before_allocation, bool
    ) or not isinstance(allocation_concealment_procedure_declared, bool):
        raise LearnerEffectStudyError("governance declarations must be booleans")
    secret = token_secret if token_secret is not None else secrets.token_bytes(32)
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise LearnerEffectStudyError("token_secret must contain at least 16 private bytes")
    generated = generated_at_utc or _utc_now()
    core: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
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
        },
        "outcomes": {
            "outcome_source": "independent_learner_assessment",
            "primary_outcome": "post_score_adjusted_for_pre_score",
            "primary_model": "participant_level_ols_post_on_arm_and_pre",
            "effect_measure": "adjusted_standardized_mean_difference",
            "standardizer": "sample_standard_deviation_of_all_randomized_pre_scores",
            "pre_score_required_for_every_randomized_participant": True,
            "missing_post_policy": "baseline_carried_forward_post_equals_pre",
            "secondary_outcome": "retention_score_descriptive_only",
            "score_minimum": float(score_minimum),
            "score_maximum": float(score_maximum),
            "grader_blinding_required": True,
            "internal_skill_score_as_outcome_prohibited": True,
        },
        "analysis_plan": {
            "estimand": "intention_to_treat_by_randomized_cluster_assignment",
            "cluster_bootstrap": "arm_stratified_resample_clusters_with_replacement",
            "confidence_level": 0.95,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "minimum_valid_bootstrap_fraction": 0.9,
            "minimum_cluster_count": minimum_cluster_count,
            "minimum_participant_count": 30,
            "minimum_primary_outcome_coverage": float(minimum_primary_coverage),
            "maximum_attrition_fraction": float(maximum_attrition_fraction),
            "minimum_educationally_meaningful_standardized_effect": float(
                minimum_meaningful_effect
            ),
            "no_test_outcome_tuning": True,
            "all_randomized_tokens_must_be_present_in_outcome_table": True,
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
    for cluster_index, arm in enumerate(assignments, start=1):
        cluster_token = _token(secret, f"{study_id}:cluster:{cluster_index}")
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
                    "retention_score": "",
                    "primary_missing_reason": "",
                    "retention_missing_reason": "",
                    "grader_blind": "",
                    "adverse_event_reported": "",
                    "protocol_deviation": "",
                }
            )
    allocation_text = _csv_text(ALLOCATION_FIELDS, allocation_rows)
    outcome_text = _csv_text(OUTCOME_FIELDS, outcome_rows)
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
        },
        "learner_effectiveness_established": False,
    }
    output = ensure_private_directory(output_directory).resolve()
    preregistration_path = write_json(output / "preregistration.json", preregistration)
    allocation_path = write_text(output / "participant_allocation.csv", allocation_text)
    outcome_path = write_text(output / "outcome_collection_template.csv", outcome_text)
    return {
        "mode": "learner_effect_study_templates_generated",
        "output_directory": str(output),
        "preregistration_path": str(preregistration_path),
        "allocation_path": str(allocation_path),
        "outcome_template_path": str(outcome_path),
        "preregistration_core_sha256": core_sha256,
        "planned_cluster_count": cluster_count,
        "planned_participant_count": len(allocation_rows),
        "identity_fields_included": False,
        "real_participant_data_generated": False,
        "learner_effectiveness_established": False,
        "external_governance_signature_included": False,
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
        raise LearnerEffectStudyError("unsupported learner-effect preregistration schema")
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
    if not isinstance(bindings, dict) or set(bindings) != {
        "allocation_csv_sha256",
        "outcome_template_csv_sha256",
    } or any(not SHA256_RE.fullmatch(str(item)) for item in bindings.values()):
        raise LearnerEffectStudyError("preregistration artifact bindings are malformed")
    return value, core, core_sha256


def _read_csv_exact(path: Path, expected_fields: tuple[str, ...], purpose: str) -> list[dict[str, str]]:
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
    design = np.column_stack(
        (np.ones(len(pre), dtype=np.float64), treatment, pre)
    )
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
        raise LearnerEffectStudyError("cluster trial requires at least one cluster per arm")
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_indices: list[int] = []
        for arm in (0, 1):
            arm_clusters = clusters_by_arm[arm]
            sampled = generator.integers(
                0, len(arm_clusters), size=len(arm_clusters)
            )
            for selected in sampled:
                sampled_indices.extend(indices_by_cluster[arm_clusters[int(selected)]])
        indices = np.asarray(sampled_indices, dtype=np.int64)
        try:
            estimate = _ols_treatment_effect(
                pre[indices], post[indices], treatment[indices]
            ) / standardizer
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
        raise LearnerEffectStudyError("allocation CSV hash differs from preregistration")
    allocation_rows = _read_csv_exact(
        allocation_file, ALLOCATION_FIELDS, "allocation"
    )
    outcome_rows = _read_csv_exact(outcomes_file, OUTCOME_FIELDS, "outcome")
    design = core.get("design")
    governance = core.get("governance")
    outcomes = core.get("outcomes")
    plan = core.get("analysis_plan")
    if not all(isinstance(value, dict) for value in (design, governance, outcomes, plan)):
        raise LearnerEffectStudyError("preregistration core sections are malformed")
    assert isinstance(design, dict)
    assert isinstance(governance, dict)
    assert isinstance(outcomes, dict)
    assert isinstance(plan, dict)
    if (
        design.get("study_design") != "cluster_randomized_controlled_trial"
        or design.get("cluster_unit") not in {"teacher", "classroom"}
        or outcomes.get("primary_outcome")
        != "post_score_adjusted_for_pre_score"
        or outcomes.get("primary_model")
        != "participant_level_ols_post_on_arm_and_pre"
        or outcomes.get("internal_skill_score_as_outcome_prohibited") is not True
        or plan.get("estimand")
        != "intention_to_treat_by_randomized_cluster_assignment"
        or plan.get("cluster_bootstrap")
        != "arm_stratified_resample_clusters_with_replacement"
    ):
        raise LearnerEffectStudyError("unsupported or altered preregistered analysis")
    expected_participants = int(design.get("planned_participant_count", -1))
    expected_clusters = int(design.get("planned_cluster_count", -1))
    if len(allocation_rows) != expected_participants:
        raise LearnerEffectStudyError("allocation row count differs from preregistration")
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
            raise LearnerEffectStudyError("allocation CSV has an invalid or duplicate row")
        prior_arm = cluster_arms.setdefault(cluster, arm)
        if prior_arm != arm:
            raise LearnerEffectStudyError("one cluster is assigned to both arms")
        allocation_by_token[participant] = row
    if len(cluster_arms) != expected_clusters:
        raise LearnerEffectStudyError("observed allocation cluster count differs from plan")
    if abs(list(cluster_arms.values()).count("A") - list(cluster_arms.values()).count("B")) > 0:
        raise LearnerEffectStudyError("allocation is not the preregistered balanced cluster trial")

    score_minimum = float(outcomes["score_minimum"])
    score_maximum = float(outcomes["score_maximum"])
    outcome_by_token: dict[str, dict[str, str]] = {}
    pre_values: list[float] = []
    post_itt_values: list[float] = []
    treatment_values: list[int] = []
    cluster_values: list[str] = []
    observed_post_count = 0
    observed_retention_count = 0
    grader_blind_count = 0
    adverse_event_yes_count = 0
    protocol_deviation_count = 0
    primary_missing_reasons: dict[str, int] = {}
    retention_missing_reasons: dict[str, int] = {}
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
        retention = _optional_score(
            row["retention_score"],
            row["retention_missing_reason"],
            name="retention_score",
            minimum=score_minimum,
            maximum=score_maximum,
        )
        grader_blind = row["grader_blind"].strip().casefold()
        adverse = row["adverse_event_reported"].strip().casefold()
        if grader_blind not in {"true", "false"}:
            raise LearnerEffectStudyError("grader_blind must be explicitly true or false")
        if adverse not in {"yes", "no"}:
            raise LearnerEffectStudyError(
                "adverse_event_reported must be explicitly yes or no"
            )
        protocol_deviation = row["protocol_deviation"].strip()
        if protocol_deviation not in PROTOCOL_DEVIATIONS:
            raise LearnerEffectStudyError(
                "protocol_deviation must use one fixed non-identifying category"
            )
        outcome_by_token[participant] = row
        pre_values.append(pre)
        post_itt_values.append(pre if post is None else post)
        treatment_values.append(int(allocation["assigned_arm"] == "A"))
        cluster_values.append(allocation["cluster_token"])
        observed_post_count += int(post is not None)
        observed_retention_count += int(retention is not None)
        grader_blind_count += int(grader_blind == "true")
        adverse_event_yes_count += int(adverse == "yes")
        protocol_deviation_count += int(protocol_deviation != "none")
        primary_reason = row["primary_missing_reason"].strip()
        retention_reason = row["retention_missing_reason"].strip()
        primary_missing_reasons[primary_reason] = (
            primary_missing_reasons.get(primary_reason, 0) + 1
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
    participant_count = len(allocation_rows)
    primary_coverage = observed_post_count / participant_count
    primary_attrition = 1.0 - primary_coverage
    retention_coverage = observed_retention_count / participant_count
    ci_lower, ci_upper = bootstrap["percentile_95_ci"]
    ethics_id = str(governance.get("ethics_approval_id", "")).strip()
    real_data = governance.get("real_participant_data_declared") is True and governance.get(
        "data_origin"
    ) == "real"
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
        "attrition_within_preregistered_limit": primary_attrition
        <= float(plan["maximum_attrition_fraction"]),
        "all_randomized_participants_in_itt": len(outcome_by_token)
        == participant_count,
        "all_graders_blind": grader_blind_count == participant_count,
        "adverse_event_reporting_complete": True,
        "cluster_aware_bootstrap_valid": bootstrap["valid_fraction"] >= 0.9,
        "effect_exceeds_preregistered_minimum": standardized_effect
        > float(plan["minimum_educationally_meaningful_standardized_effect"]),
        "confidence_interval_lower_exceeds_preregistered_minimum": ci_lower
        > float(plan["minimum_educationally_meaningful_standardized_effect"]),
        "internal_skill_score_used_as_learning_outcome": False,
    }
    local_positive_gate = all(
        value for key, value in gates.items() if key != "internal_skill_score_used_as_learning_outcome"
    ) and gates["internal_skill_score_used_as_learning_outcome"] is False
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
        "external_evidence_signature_verified": False,
    }
    result = {
        "schema": ANALYSIS_SCHEMA,
        "study_id": core["study_id"],
        "input_bindings": {
            "preregistration_sha256": _file_sha256(prereg_path),
            "preregistration_core_sha256": core_sha256,
            "allocation_csv_sha256": _file_sha256(allocation_file),
            "completed_outcomes_csv_sha256": _file_sha256(outcomes_file),
            "analysis_code_sha256": _file_sha256(Path(__file__).resolve()),
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
        "coverage_and_attrition": {
            "eligible_randomized_participant_count": participant_count,
            "observed_primary_outcome_count": observed_post_count,
            "baseline_carried_forward_count": participant_count
            - observed_post_count,
            "primary_outcome_coverage_fraction": round(primary_coverage, 6),
            "primary_attrition_fraction": round(primary_attrition, 6),
            "retention_outcome_coverage_fraction": round(retention_coverage, 6),
            "primary_missing_reason_counts": primary_missing_reasons,
            "retention_missing_reason_counts": retention_missing_reasons,
        },
        "blinding_and_safety": {
            "grader_blind_count": grader_blind_count,
            "adverse_event_yes_count": adverse_event_yes_count,
            "adverse_event_reporting_complete": True,
            "protocol_deviation_count": protocol_deviation_count,
        },
        "gates": gates,
        "local_preregistered_positive_result_gate": local_positive_gate,
        "eligible_for_external_governance_review": local_positive_gate,
        "learner_effect_establishment_gates": establishment_gates,
        "learner_effect_establishment_gate": all(establishment_gates.values()),
        "learner_effectiveness_established": False,
        "claim_boundary": {
            "internal_skill_scores_are_learning_effectiveness": False,
            "row_level_tokens_or_scores_in_result": False,
            "self_declared_ethics_or_data_origin_independently_verified": False,
            "external_governance_signature_verified": False,
            "existing_external_evidence_chain_still_required": True,
            "valid_claim": (
                "local preregistered cluster-aware learner-study analysis; not "
                "established learning-effect evidence without external governance"
            ),
        },
        "external_evidence_handoff": {
            "evidence_kind": "real_learner_effectiveness",
            "requires_independent_p_value_and_complete_registered_reporting": True,
            "requires_prepare_sign_verify_external_research_evidence_commands": True,
            "analysis_result_file_sha256_must_be_computed_by_external_preparer": True,
        },
        "software": {
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


def write_learner_effect_analysis(
    output_path: str | Path,
    result: dict[str, Any],
) -> Path:
    """Write a private aggregate analysis and bind the handoff to exact content."""

    if result.get("schema") != ANALYSIS_SCHEMA:
        raise LearnerEffectStudyError("not a learner-effect analysis result")
    value = json.loads(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return write_json(output_path, value)
