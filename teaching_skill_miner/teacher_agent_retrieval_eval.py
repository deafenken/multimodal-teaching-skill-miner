"""Gold-bound evaluation for teacher-resource retrieval.

The evaluator keeps relevance judgments outside the retrieval call.  A
retriever only sees the query and teacher resources; relevant chunk IDs,
forbidden visual chunks, and expected conflict outcomes are consumed after the
receipt has been produced.  This prevents the development evaluator from
turning benchmark gold into model-visible teaching context.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
import math
import re
from typing import Any

from .teacher_agent_resource_retrieval import RESOURCE_RETRIEVAL_SCHEMA


RETRIEVAL_EVAL_SCHEMA = "teaching_skill_miner.resource_retrieval_evaluation.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CONSISTENCY = frozenset(
    {
        "not_verified",
        "query_supported",
        "query_contradicted",
        "conflicting_sources",
        "uncertain",
    }
)
_MODALITIES = frozenset(
    {"text", "table", "chart", "equation", "mixed", "contradiction"}
)


class ResourceRetrievalEvaluationError(ValueError):
    """Raised when an evaluation case or retrieval receipt is malformed."""


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _safe_id(value: Any, *, field: str) -> str:
    text = str(value)
    if _SAFE_ID.fullmatch(text) is None:
        raise ResourceRetrievalEvaluationError(f"{field} is invalid")
    return text


def _chunk_ids(value: Any, *, field: str, required: bool) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResourceRetrievalEvaluationError(f"{field} must be an array")
    result: list[str] = []
    for raw in value:
        chunk_id = _safe_id(raw, field=field)
        if chunk_id in result:
            raise ResourceRetrievalEvaluationError(f"{field} contains duplicates")
        result.append(chunk_id)
    if required and not result:
        raise ResourceRetrievalEvaluationError(f"{field} cannot be empty")
    if len(result) > 64:
        raise ResourceRetrievalEvaluationError(f"{field} exceeds its bound")
    return result


def _validated_case(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResourceRetrievalEvaluationError("retrieval case must be an object")
    required = {
        "case_id",
        "query",
        "relevant_chunk_ids",
        "forbidden_chunk_ids",
        "expected_consistency",
        "k",
    }
    optional = {
        "modality",
        "relevance_grades",
        "entailing_chunk_ids",
        "contradicting_chunk_ids",
        "expected_abstain",
    }
    if not required.issubset(value) or not set(value).issubset(required | optional):
        raise ResourceRetrievalEvaluationError("retrieval case fields are invalid")
    query = str(value["query"]).strip()
    if not query or len(query) > 300:
        raise ResourceRetrievalEvaluationError("retrieval case query is invalid")
    expected = str(value["expected_consistency"])
    if expected not in _CONSISTENCY:
        raise ResourceRetrievalEvaluationError(
            "retrieval case consistency target is invalid"
        )
    k = value["k"]
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        raise ResourceRetrievalEvaluationError("retrieval case k is invalid")
    relevant = _chunk_ids(
        value["relevant_chunk_ids"], field="relevant_chunk_ids", required=True
    )
    forbidden = _chunk_ids(
        value["forbidden_chunk_ids"], field="forbidden_chunk_ids", required=False
    )
    if set(relevant) & set(forbidden):
        raise ResourceRetrievalEvaluationError(
            "relevant and forbidden chunks must be disjoint"
        )
    modality = str(value.get("modality", "text"))
    if modality not in _MODALITIES:
        raise ResourceRetrievalEvaluationError("retrieval case modality is invalid")
    raw_grades = value.get("relevance_grades")
    if raw_grades is None:
        relevance_grades = {chunk_id: 1 for chunk_id in relevant}
    else:
        if not isinstance(raw_grades, Mapping) or set(raw_grades) != set(relevant):
            raise ResourceRetrievalEvaluationError(
                "retrieval case relevance grades must exactly cover relevant chunks"
            )
        relevance_grades: dict[str, int] = {}
        for raw_chunk_id, raw_grade in raw_grades.items():
            chunk_id = _safe_id(raw_chunk_id, field="relevance_grades")
            if (
                isinstance(raw_grade, bool)
                or not isinstance(raw_grade, int)
                or not 1 <= raw_grade <= 3
            ):
                raise ResourceRetrievalEvaluationError(
                    "retrieval case relevance grade is invalid"
                )
            relevance_grades[chunk_id] = raw_grade
    entailing = _chunk_ids(
        value.get("entailing_chunk_ids", []),
        field="entailing_chunk_ids",
        required=False,
    )
    contradicting = _chunk_ids(
        value.get("contradicting_chunk_ids", []),
        field="contradicting_chunk_ids",
        required=False,
    )
    if (
        set(entailing) & set(contradicting)
        or not (set(entailing) | set(contradicting)).issubset(set(relevant))
    ):
        raise ResourceRetrievalEvaluationError(
            "retrieval case claim relation labels are invalid"
        )
    expected_abstain = value.get(
        "expected_abstain", expected != "query_supported"
    )
    if not isinstance(expected_abstain, bool):
        raise ResourceRetrievalEvaluationError(
            "retrieval case expected abstention is invalid"
        )
    return {
        "case_id": _safe_id(value["case_id"], field="case_id"),
        "query": query,
        "relevant_chunk_ids": relevant,
        "forbidden_chunk_ids": forbidden,
        "expected_consistency": expected,
        "expected_abstain": expected_abstain,
        "modality": modality,
        "relevance_grades": relevance_grades,
        "entailing_chunk_ids": entailing,
        "contradicting_chunk_ids": contradicting,
        "k": k,
    }


def _validated_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != RESOURCE_RETRIEVAL_SCHEMA
        or not isinstance(value.get("results"), list)
        or value.get("result_count") != len(value["results"])
        or value.get("grading_evidence_allowed") is not False
        or value.get("claim_consistency_status") not in _CONSISTENCY
        or not isinstance(value.get("safe_to_synthesize"), bool)
    ):
        raise ResourceRetrievalEvaluationError("retrieval receipt contract is invalid")
    result = deepcopy(dict(value))
    citation_ids: list[str] = []
    for row in result["results"]:
        provenance = row.get("provenance", {}) if isinstance(row, Mapping) else {}
        excerpt = row.get("excerpt") if isinstance(row, Mapping) else None
        excerpt_digest = (
            provenance.get("excerpt_content_sha256")
            if isinstance(provenance, Mapping)
            else None
        )
        if excerpt_digest is None and isinstance(provenance, Mapping):
            # Compatibility with v1 receipts that never truncated a chunk.
            excerpt_digest = provenance.get("chunk_content_sha256")
        if (
            not isinstance(excerpt, str)
            or not excerpt
            or not isinstance(provenance, Mapping)
            or _SAFE_ID.fullmatch(str(provenance.get("chunk_id", ""))) is None
            or _DIGEST.fullmatch(str(provenance.get("chunk_content_sha256", "")))
            is None
            or _DIGEST.fullmatch(str(excerpt_digest or "")) is None
            or sha256(excerpt.encode("utf-8")).hexdigest()
            != excerpt_digest
            or provenance.get("content_is_untrusted_instruction_data") is not True
            or row.get("eligible_for_grading") is not False
        ):
            raise ResourceRetrievalEvaluationError(
                "retrieval result provenance integrity is invalid"
            )
        if provenance.get("excerpt_is_partial") is False and (
            provenance.get("chunk_content_sha256") != excerpt_digest
        ):
            raise ResourceRetrievalEvaluationError(
                "retrieval result full-chunk hash binding is invalid"
            )
        if provenance.get("needs_visual_review") is True and row.get(
            "eligible_for_synthesis"
        ) is not False:
            raise ResourceRetrievalEvaluationError(
                "visual-review-pending result cannot be synthesis evidence"
            )
        citation_id = row.get("citation_id")
        if citation_id is not None:
            citation_ids.append(_safe_id(citation_id, field="citation_id"))
    if len(set(citation_ids)) != len(citation_ids):
        raise ResourceRetrievalEvaluationError(
            "retrieval result citation IDs must be unique"
        )
    claim_trace = result.get("claim_trace")
    if claim_trace is not None:
        if (
            not isinstance(claim_trace, Mapping)
            or claim_trace.get("claim_sha256") != result.get("query_sha256")
            or claim_trace.get("consistency_status")
            != result.get("claim_consistency_status")
            or not isinstance(claim_trace.get("citations"), list)
            or [item.get("citation_id") for item in claim_trace["citations"]]
            != citation_ids
            or result.get("claim_trace_sha256") != _canonical_sha256(claim_trace)
        ):
            raise ResourceRetrievalEvaluationError(
                "retrieval claim-to-chunk trace is invalid"
            )
        for trace, row in zip(claim_trace["citations"], result["results"], strict=True):
            provenance = row["provenance"]
            if (
                trace.get("chunk_id") != provenance.get("chunk_id")
                or trace.get("chunk_content_sha256")
                != provenance.get("chunk_content_sha256")
                or trace.get("excerpt_content_sha256")
                != provenance.get("excerpt_content_sha256")
                or trace.get("relation") != row.get("claim_relation")
            ):
                raise ResourceRetrievalEvaluationError(
                    "retrieval claim citation binding is invalid"
                )
    receipt_digest = result.get("receipt_sha256")
    if receipt_digest is not None:
        unsigned = deepcopy(result)
        unsigned.pop("receipt_sha256", None)
        if (
            _DIGEST.fullmatch(str(receipt_digest)) is None
            or receipt_digest != _canonical_sha256(unsigned)
        ):
            raise ResourceRetrievalEvaluationError(
                "retrieval deterministic receipt hash is invalid"
            )
    return result


def score_resource_retrieval_predictions(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score hidden relevance/conflict labels against retrieval receipts."""

    if not cases:
        raise ResourceRetrievalEvaluationError("retrieval evaluation needs cases")
    validated_cases = [_validated_case(item) for item in cases]
    if len({item["case_id"] for item in validated_cases}) != len(validated_cases):
        raise ResourceRetrievalEvaluationError("retrieval case IDs must be unique")
    if set(predictions) != {item["case_id"] for item in validated_cases}:
        raise ResourceRetrievalEvaluationError(
            "retrieval prediction IDs must exactly match cases"
        )

    rows: list[dict[str, Any]] = []
    for case in validated_cases:
        receipt = _validated_receipt(predictions[case["case_id"]])
        if receipt.get("query_sha256") != sha256(case["query"].encode()).hexdigest():
            raise ResourceRetrievalEvaluationError(
                "retrieval receipt query hash does not match its hidden case"
            )
        top = receipt["results"][: case["k"]]
        predicted = [str(item["provenance"]["chunk_id"]) for item in top]
        relevant = set(case["relevant_chunk_ids"])
        hits = [chunk_id in relevant for chunk_id in predicted]
        recall = sum(hits) / len(relevant)
        precision = sum(hits) / len(predicted) if predicted else 0.0
        grades = case["relevance_grades"]
        predicted_grades = [grades.get(chunk_id, 0) for chunk_id in predicted]
        dcg = sum(
            ((2.0**grade - 1.0) / math.log2(rank + 2))
            for rank, grade in enumerate(predicted_grades)
            if grade
        )
        ideal_grades = sorted(grades.values(), reverse=True)[: case["k"]]
        ideal_dcg = sum(
            (2.0**grade - 1.0) / math.log2(rank + 2)
            for rank, grade in enumerate(ideal_grades)
        )
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        forbidden = set(case["forbidden_chunk_ids"])
        forbidden_hit_count = sum(chunk_id in forbidden for chunk_id in predicted)
        expected_consistency = case["expected_consistency"]
        consistency_correct = (
            receipt["claim_consistency_status"] == expected_consistency
        )
        abstention_correct = (
            receipt["safe_to_synthesize"] is False
            if case["expected_abstain"]
            else receipt["safe_to_synthesize"] is True
        )
        predicted_relations = {
            str(item["provenance"]["chunk_id"]): str(
                item.get("claim_relation", "not_verified")
            )
            for item in top
        }
        relation_labels = {
            **{chunk_id: "entails" for chunk_id in case["entailing_chunk_ids"]},
            **{
                chunk_id: "contradicts"
                for chunk_id in case["contradicting_chunk_ids"]
            },
        }
        relation_correct = sum(
            predicted_relations.get(chunk_id) == expected_relation
            for chunk_id, expected_relation in relation_labels.items()
        )
        relation_accuracy = (
            relation_correct / len(relation_labels) if relation_labels else None
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "modality": case["modality"],
                "query_sha256": sha256(case["query"].encode()).hexdigest(),
                "recall_at_k": round(recall, 6),
                "precision_at_k": round(precision, 6),
                "citation_precision_at_k": round(precision, 6),
                "ndcg_at_k": round(ndcg, 6),
                "citation_integrity": True,
                "claim_trace_integrity": receipt.get("claim_trace") is not None,
                "forbidden_chunk_hit_count": forbidden_hit_count,
                "consistency_correct": consistency_correct,
                "abstention_correct": abstention_correct,
                "entailment_relation_accuracy": (
                    round(relation_accuracy, 6)
                    if relation_accuracy is not None
                    else None
                ),
            }
        )
    count = len(rows)
    relation_rows = [
        item
        for item in rows
        if item["entailment_relation_accuracy"] is not None
    ]
    modality_metrics: dict[str, dict[str, Any]] = {}
    for modality in sorted({str(item["modality"]) for item in rows}):
        subset = [item for item in rows if item["modality"] == modality]
        modality_metrics[modality] = {
            "case_count": len(subset),
            "mean_recall_at_k": round(
                sum(item["recall_at_k"] for item in subset) / len(subset), 6
            ),
            "mean_ndcg_at_k": round(
                sum(item["ndcg_at_k"] for item in subset) / len(subset), 6
            ),
            "citation_precision_at_k": round(
                sum(item["citation_precision_at_k"] for item in subset)
                / len(subset),
                6,
            ),
            "abstention_accuracy": round(
                sum(item["abstention_correct"] for item in subset) / len(subset),
                6,
            ),
            "entailment_relation_accuracy": (
                round(
                    sum(
                        item["entailment_relation_accuracy"]
                        for item in subset
                        if item["entailment_relation_accuracy"] is not None
                    )
                    / sum(
                        item["entailment_relation_accuracy"] is not None
                        for item in subset
                    ),
                    6,
                )
                if any(
                    item["entailment_relation_accuracy"] is not None
                    for item in subset
                )
                else None
            ),
        }
    report = {
        "schema": RETRIEVAL_EVAL_SCHEMA,
        "case_count": count,
        "mean_recall_at_k": round(sum(item["recall_at_k"] for item in rows) / count, 6),
        "mean_precision_at_k": round(
            sum(item["precision_at_k"] for item in rows) / count, 6
        ),
        "citation_precision_at_k": round(
            sum(item["citation_precision_at_k"] for item in rows) / count, 6
        ),
        "mean_ndcg_at_k": round(sum(item["ndcg_at_k"] for item in rows) / count, 6),
        "citation_integrity_rate": 1.0,
        "claim_trace_integrity_rate": round(
            sum(item["claim_trace_integrity"] for item in rows) / count, 6
        ),
        "forbidden_chunk_hit_rate": round(
            sum(item["forbidden_chunk_hit_count"] > 0 for item in rows) / count,
            6,
        ),
        "consistency_accuracy": round(
            sum(item["consistency_correct"] for item in rows) / count, 6
        ),
        "contradiction_abstention_accuracy": round(
            sum(item["abstention_correct"] for item in rows) / count, 6
        ),
        "abstention_accuracy": round(
            sum(item["abstention_correct"] for item in rows) / count, 6
        ),
        "entailment_relation_accuracy": (
            round(
                sum(item["entailment_relation_accuracy"] for item in relation_rows)
                / len(relation_rows),
                6,
            )
            if relation_rows
            else None
        ),
        "modality_metrics": modality_metrics,
        "cases": rows,
        "student_evidence_used": False,
        "benchmark_gold_exposed_to_retriever": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def run_resource_retrieval_evaluation(
    cases: Sequence[Mapping[str, Any]],
    retrieve: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run a retriever with queries only, then score against hidden judgments."""

    validated = [_validated_case(item) for item in cases]
    predictions = {
        case["case_id"]: deepcopy(dict(retrieve(case["query"]))) for case in validated
    }
    return score_resource_retrieval_predictions(validated, predictions)


__all__ = [
    "RETRIEVAL_EVAL_SCHEMA",
    "ResourceRetrievalEvaluationError",
    "run_resource_retrieval_evaluation",
    "score_resource_retrieval_predictions",
]
