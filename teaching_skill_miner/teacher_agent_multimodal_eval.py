"""Gold-hidden evaluation for visual teaching evidence boundaries.

The analyzer receives only media bytes, MIME type, and task context.  Expected
modality/claims/abstentions remain hidden until after the evidence record is
returned.  This evaluates semantic traceability and safety, not student
learning effectiveness or broad vision-model accuracy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import base64
import binascii
import hashlib
import json
import math
import re
from typing import Any


MULTIMODAL_EVAL_SCHEMA = "teaching_skill_miner.multimodal_eval.v1"
MULTIMODAL_LOCKBOX_INPUTS_SCHEMA = "teaching_skill_miner.multimodal_lockbox_inputs.v1"
MULTIMODAL_LOCKBOX_GOLD_SCHEMA = "teaching_skill_miner.multimodal_lockbox_gold.v1"
MULTIMODAL_LOCKBOX_REPORT_SCHEMA = "teaching_skill_miner.multimodal_lockbox_report.v1"
MAX_MULTIMODAL_EVAL_CASES = 500
REQUIRED_MATURITY_COVERAGE = frozenset(
    {
        "scanned_pdf",
        "table",
        "chart",
        "geometry",
        "handwriting",
        "formula",
        "slide_notes_conflict",
        "audio",
        "video",
    }
)


class MultimodalEvalError(ValueError):
    """Raised when a hidden visual evaluation fixture is malformed."""


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in str(value).replace("，", " ").replace("。", " ").split()
        if token.strip()
    }


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
        raise MultimodalEvalError("multimodal lockbox is not canonical JSON") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _lockbox_cases(
    inputs: Mapping[str, Any], gold: Mapping[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any], bytes]]:
    if inputs.get("schema") != MULTIMODAL_LOCKBOX_INPUTS_SCHEMA:
        raise MultimodalEvalError("multimodal lockbox input schema is invalid")
    if gold.get("schema") != MULTIMODAL_LOCKBOX_GOLD_SCHEMA:
        raise MultimodalEvalError("multimodal lockbox gold schema is invalid")
    dataset_id = inputs.get("dataset_id")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or gold.get("dataset_id") != dataset_id
        or gold.get("inputs_sha256") != _sha(inputs)
    ):
        raise MultimodalEvalError("multimodal lockbox input/gold binding is invalid")
    input_cases = inputs.get("cases")
    gold_cases = gold.get("cases")
    if (
        not isinstance(input_cases, list)
        or not 1 <= len(input_cases) <= MAX_MULTIMODAL_EVAL_CASES
        or not isinstance(gold_cases, list)
        or len(gold_cases) != len(input_cases)
    ):
        raise MultimodalEvalError("multimodal lockbox cases are invalid")
    paired: list[tuple[dict[str, Any], dict[str, Any], bytes]] = []
    seen: set[str] = set()
    coverage: set[str] = set()
    for index, (input_raw, gold_raw) in enumerate(zip(input_cases, gold_cases)):
        if not isinstance(input_raw, Mapping) or not isinstance(gold_raw, Mapping):
            raise MultimodalEvalError(f"multimodal lockbox case {index} is invalid")
        input_case = dict(input_raw)
        gold_case = dict(gold_raw)
        case_id = input_case.get("case_id")
        mime_type = input_case.get("mime_type")
        context = input_case.get("task_context")
        encoded = input_case.get("media_base64")
        if (
            not isinstance(case_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,79}", case_id)
            or case_id in seen
            or gold_case.get("case_id") != case_id
            or not isinstance(mime_type, str)
            or not mime_type
            or not isinstance(context, str)
            or not context.strip()
            or len(context) > 1_200
            or not isinstance(encoded, str)
            or not encoded
        ):
            raise MultimodalEvalError(f"multimodal lockbox case {index} is invalid")
        try:
            media = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MultimodalEvalError(
                f"multimodal lockbox case {index} media is invalid"
            ) from exc
        if not media or len(media) > 32 * 1024 * 1024 or input_case.get(
            "media_sha256"
        ) != hashlib.sha256(media).hexdigest():
            raise MultimodalEvalError(
                f"multimodal lockbox case {index} media binding is invalid"
            )
        coverage_tag = gold_case.get("coverage_tag")
        expected_source = gold_case.get("expected_source_modality")
        expected_semantic = gold_case.get("expected_semantic_modality")
        expected_terms = gold_case.get("expected_claim_terms")
        required_decision = gold_case.get("required_decision")
        privacy_scope = gold_case.get("privacy_scope")
        if (
            coverage_tag not in REQUIRED_MATURITY_COVERAGE
            or not isinstance(expected_source, str)
            or not expected_source
            or not isinstance(expected_semantic, str)
            or not expected_semantic
            or not isinstance(expected_terms, list)
            or any(not isinstance(term, str) or not term for term in expected_terms)
            or required_decision not in {"observation", "requires_confirmation", "abstain"}
            or privacy_scope not in {"local", "remote"}
            or not isinstance(gold_case.get("temporal_provenance_required"), bool)
            or not isinstance(gold_case.get("conflict_required"), bool)
        ):
            raise MultimodalEvalError(
                f"multimodal lockbox gold case {index} is invalid"
            )
        seen.add(case_id)
        coverage.add(str(coverage_tag))
        paired.append((input_case, gold_case, media))
    if coverage != REQUIRED_MATURITY_COVERAGE:
        missing = ",".join(sorted(REQUIRED_MATURITY_COVERAGE - coverage))
        extra = ",".join(sorted(coverage - REQUIRED_MATURITY_COVERAGE))
        raise MultimodalEvalError(
            f"multimodal lockbox maturity coverage is incomplete: missing={missing}; extra={extra}"
        )
    return paired


def _boundary_held(evidence: Mapping[str, Any]) -> bool:
    transcription = evidence.get("transcription")
    semantic = evidence.get("semantic_analysis")
    return bool(
        evidence.get("transcription_is_semantic_understanding") is False
        and evidence.get("semantic_analysis_is_answer_correctness") is False
        and evidence.get("grading_evidence_allowed") is False
        and evidence.get("mastery_evidence_allowed") is False
        and isinstance(transcription, Mapping)
        and transcription.get("grading_evidence_allowed") is False
        and transcription.get("mastery_evidence_allowed") is False
        and isinstance(semantic, Mapping)
        and semantic.get("grading_evidence_allowed") is False
        and semantic.get("mastery_evidence_allowed") is False
    )


def _provenance_held(evidence: Mapping[str, Any], media: bytes) -> bool:
    claims = evidence.get("claims", [])
    claim_locators = bool(
        isinstance(claims, list)
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("evidence_locator"), str)
            and bool(str(item["evidence_locator"]).strip())
            and item.get("verification_status") == "unverified_provider_observation"
            for item in claims
        )
    )
    return bool(
        evidence.get("media_sha256") == hashlib.sha256(media).hexdigest()
        and re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("provider_spec_sha256", "")))
        and re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("provider_result_sha256", "")))
        and claim_locators
    )


def _temporal_provenance_held(evidence: Mapping[str, Any]) -> bool:
    transcription = evidence.get("transcription")
    if not isinstance(transcription, Mapping):
        return False
    segments = transcription.get("segments")
    if not isinstance(segments, list) or not segments:
        return False
    previous = -1
    for segment in segments:
        if (
            not isinstance(segment, Mapping)
            or isinstance(segment.get("start_ms"), bool)
            or not isinstance(segment.get("start_ms"), int)
            or isinstance(segment.get("end_ms"), bool)
            or not isinstance(segment.get("end_ms"), int)
            or segment["start_ms"] < previous
            or segment["end_ms"] <= segment["start_ms"]
            or not str(segment.get("evidence_locator", "")).strip()
        ):
            return False
        previous = segment["start_ms"]
    return True


def _privacy_held(evidence: Mapping[str, Any], scope: str) -> bool:
    receipt = evidence.get("privacy_receipt")
    if not isinstance(receipt, Mapping):
        return False
    if scope == "local":
        return bool(
            evidence.get("remote_media_sent") is False
            and receipt.get("execution_scope") == "local"
            and receipt.get("raw_media_sent") is False
            and receipt.get("provider_retention_days") == 0
            and receipt.get("consent_id") is None
            and receipt.get("consent_receipt_sha256") is None
        )
    return bool(
        evidence.get("remote_media_sent") is True
        and receipt.get("execution_scope") == "remote"
        and receipt.get("raw_media_sent") is True
        and isinstance(receipt.get("processing_region"), str)
        and isinstance(receipt.get("provider_retention_days"), int)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("consent_receipt_sha256", ""))
        )
        and receipt.get("data_categories") == ["learner_image"]
    )


def _raw_media_not_exposed(evidence: Mapping[str, Any], media: bytes) -> bool:
    serialized = repr(evidence)
    forbidden = {media.hex(), base64.b64encode(media).decode("ascii")}
    try:
        decoded = media.decode("utf-8")
    except UnicodeDecodeError:
        decoded = ""
    if len(decoded) >= 4 and all(
        character.isprintable() or character.isspace() for character in decoded
    ):
        forbidden.add(decoded)
        forbidden.add(repr(decoded)[1:-1])
    return all(value not in serialized for value in forbidden if value)


def evaluate_multimodal_lockbox(
    inputs: Mapping[str, Any],
    gold: Mapping[str, Any],
    analyzer: Callable[[bytes, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the deterministic, gold-hidden maturity lockbox without a network."""

    if not isinstance(inputs, Mapping) or not isinstance(gold, Mapping):
        raise MultimodalEvalError("multimodal lockbox documents are invalid")
    paired = _lockbox_cases(inputs, gold)
    rows: list[dict[str, Any]] = []
    for input_case, gold_case, media in paired:
        try:
            raw = analyzer(
                media,
                str(input_case["mime_type"]),
                str(input_case["task_context"]),
            )
        except Exception as exc:  # noqa: BLE001 - providers are scored fail-closed.
            raw = {"error_type": type(exc).__name__}
        evidence = deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
        semantic = evidence.get("semantic_analysis", {})
        description = (
            str(semantic.get("description", ""))
            if isinstance(semantic, Mapping)
            else ""
        )
        claims = evidence.get("claims", [])
        claim_text = " ".join(
            str(item.get("statement", ""))
            for item in claims
            if isinstance(item, Mapping)
        )
        expected_terms = _tokens(" ".join(gold_case["expected_claim_terms"]))
        observed_terms = _tokens(claim_text + " " + description)
        recall = (
            len(expected_terms & observed_terms) / len(expected_terms)
            if expected_terms
            else 1.0
        )
        conflicts = evidence.get("conflicts", [])
        conflict_held = bool(isinstance(conflicts, list) and conflicts)
        decision = evidence.get("decision")
        decision_held = decision == gold_case["required_decision"]
        if gold_case["conflict_required"]:
            decision_held = decision_held and conflict_held and decision in {
                "requires_confirmation",
                "abstain",
            }
        temporal_held = (
            _temporal_provenance_held(evidence)
            if gold_case["temporal_provenance_required"]
            else True
        )
        rows.append(
            {
                "case_id": input_case["case_id"],
                "coverage_tag": gold_case["coverage_tag"],
                "source_modality_correct": evidence.get("source_modality")
                == gold_case["expected_source_modality"],
                "semantic_modality_correct": evidence.get("modality")
                == gold_case["expected_semantic_modality"],
                "claim_recall": round(recall, 6),
                "layer_and_assessment_boundary_held": _boundary_held(evidence),
                "decision_and_conflict_gate_held": decision_held,
                "provider_and_locator_provenance_held": _provenance_held(
                    evidence, media
                ),
                "temporal_provenance_held": temporal_held,
                "privacy_receipt_held": _privacy_held(
                    evidence, str(gold_case["privacy_scope"])
                ),
                "raw_media_not_exposed": _raw_media_not_exposed(evidence, media),
                "analyzer_error": evidence.get("error_type"),
            }
        )
    count = len(rows)
    metric_fields = (
        "source_modality_correct",
        "semantic_modality_correct",
        "layer_and_assessment_boundary_held",
        "decision_and_conflict_gate_held",
        "provider_and_locator_provenance_held",
        "temporal_provenance_held",
        "privacy_receipt_held",
        "raw_media_not_exposed",
    )
    metrics = {
        field: round(sum(bool(row[field]) for row in rows) / count, 6)
        for field in metric_fields
    }
    metrics["mean_claim_recall"] = round(
        sum(float(row["claim_recall"]) for row in rows) / count, 6
    )
    unsigned = {
        "schema": MULTIMODAL_LOCKBOX_REPORT_SCHEMA,
        "dataset_id": inputs["dataset_id"],
        "inputs_sha256": _sha(inputs),
        "gold_sha256": _sha(gold),
        "case_count": count,
        "coverage": sorted(REQUIRED_MATURITY_COVERAGE),
        "metrics": metrics,
        "cases": rows,
        "claim_boundary": {
            "gold_hidden_from_analyzer": True,
            "network_required": False,
            "transcription_semantics_assessment_separated": True,
            "student_learning_effectiveness_established": False,
            "deployment_multimodal_accuracy_established": False,
        },
    }
    return {**unsigned, "report_sha256": _sha(unsigned)}


def evaluate_multimodal_analyzer(
    cases: Sequence[Mapping[str, Any]],
    analyzer: Callable[[bytes, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one analyzer without exposing hidden expected fields to it."""

    if (
        isinstance(cases, (str, bytes))
        or not isinstance(cases, Sequence)
        or not 1 <= len(cases) <= MAX_MULTIMODAL_EVAL_CASES
    ):
        raise MultimodalEvalError("multimodal eval cases are invalid")
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise MultimodalEvalError(f"multimodal eval case {index} is invalid")
        case_id = case.get("case_id")
        media = case.get("media")
        mime_type = case.get("mime_type")
        context = case.get("task_context")
        expected_modality = case.get("expected_modality")
        expected_claim_terms = case.get("expected_claim_terms")
        must_abstain = case.get("must_abstain")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(media, bytes)
            or not media
            or not isinstance(mime_type, str)
            or not mime_type
            or not isinstance(context, str)
            or not context.strip()
            or not isinstance(expected_modality, str)
            or not isinstance(expected_claim_terms, list)
            or any(
                not isinstance(term, str) or not term for term in expected_claim_terms
            )
            or not isinstance(must_abstain, bool)
        ):
            raise MultimodalEvalError(f"multimodal eval case {index} is invalid")
        # Deliberately pass no expected field or case metadata to the analyzer.
        try:
            raw = analyzer(media, mime_type, context)
        except Exception as exc:  # noqa: BLE001 - a model/provider can fail closed.
            raw = {"error_type": type(exc).__name__}
        evidence = deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
        claims = evidence.get("claims", [])
        claim_text = " ".join(
            str(item.get("statement", ""))
            for item in claims
            if isinstance(item, Mapping)
        )
        observed_terms = _tokens(
            claim_text + " " + str(evidence.get("description", ""))
        )
        expected_terms = _tokens(" ".join(expected_claim_terms))
        claim_recall = (
            len(expected_terms & observed_terms) / len(expected_terms)
            if expected_terms
            else 1.0
        )
        uncertainties = evidence.get("uncertainties", [])
        abstained = bool(
            isinstance(uncertainties, list)
            and uncertainties
            and evidence.get("grading_evidence_allowed") is False
            and evidence.get("requires_teacher_review") is True
        )
        raw_hash = hashlib.sha256(media).hexdigest()
        serialized = repr(evidence)
        no_raw_leak = raw_hash not in serialized and media.hex() not in serialized
        rows.append(
            {
                "case_id": case_id,
                "modality_correct": evidence.get("modality") == expected_modality,
                "claim_recall": round(claim_recall, 6),
                "abstention_correct": abstained if must_abstain else True,
                "grading_boundary_held": evidence.get("grading_evidence_allowed")
                is False,
                "raw_media_not_exposed": no_raw_leak,
                "analyzer_error": evidence.get("error_type"),
            }
        )
    count = len(rows)
    metrics = {
        "modality_accuracy": sum(row["modality_correct"] for row in rows) / count,
        "mean_claim_recall": sum(row["claim_recall"] for row in rows) / count,
        "required_abstention_rate": sum(row["abstention_correct"] for row in rows)
        / count,
        "grading_boundary_rate": sum(row["grading_boundary_held"] for row in rows)
        / count,
        "raw_media_privacy_rate": sum(row["raw_media_not_exposed"] for row in rows)
        / count,
    }
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise MultimodalEvalError("multimodal eval metric is invalid")
    return {
        "schema": MULTIMODAL_EVAL_SCHEMA,
        "case_count": count,
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "cases": rows,
        "claim_boundary": {
            "gold_hidden_from_analyzer": True,
            "visual_semantic_traceability_measured": True,
            "student_learning_effectiveness_established": False,
            "deployment_vision_accuracy_established": False,
        },
    }


__all__ = [
    "MAX_MULTIMODAL_EVAL_CASES",
    "MULTIMODAL_EVAL_SCHEMA",
    "MULTIMODAL_LOCKBOX_GOLD_SCHEMA",
    "MULTIMODAL_LOCKBOX_INPUTS_SCHEMA",
    "MULTIMODAL_LOCKBOX_REPORT_SCHEMA",
    "MultimodalEvalError",
    "REQUIRED_MATURITY_COVERAGE",
    "evaluate_multimodal_analyzer",
    "evaluate_multimodal_lockbox",
]
