from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_multimodal_eval import (
    MultimodalEvalError,
    evaluate_multimodal_analyzer,
    evaluate_multimodal_lockbox,
)


ROOT = Path(__file__).resolve().parents[1]


def test_multimodal_eval_keeps_gold_hidden_and_scores_abstention_boundary() -> None:
    received: list[tuple[bytes, str, str]] = []

    def analyzer(media: bytes, mime_type: str, context: str):
        received.append((media, mime_type, context))
        return {
            "modality": "chart",
            "description": "强光组数值 8，弱光组数值 2。",
            "claims": [
                {"statement": "强光 8 弱光 2", "evidence_locator": "chart-labels"}
            ],
            "uncertainties": ["纵轴单位不可见"],
            "grading_evidence_allowed": False,
            "requires_teacher_review": True,
        }

    report = evaluate_multimodal_analyzer(
        [
            {
                "case_id": "chart_01",
                "media": b"hidden-chart-bytes",
                "mime_type": "image/png",
                "task_context": "描述图表中可见的数值",
                "expected_modality": "chart",
                "expected_claim_terms": ["强光", "8", "弱光", "2"],
                "must_abstain": True,
            }
        ],
        analyzer,
    )

    assert received == [(b"hidden-chart-bytes", "image/png", "描述图表中可见的数值")]
    assert report["metrics"] == {
        "modality_accuracy": 1.0,
        "mean_claim_recall": 1.0,
        "required_abstention_rate": 1.0,
        "grading_boundary_rate": 1.0,
        "raw_media_privacy_rate": 1.0,
    }
    assert report["claim_boundary"]["gold_hidden_from_analyzer"] is True
    assert (
        report["claim_boundary"]["student_learning_effectiveness_established"] is False
    )


def test_multimodal_eval_records_provider_failure_as_safe_zero_not_false_success() -> (
    None
):
    def unavailable(_media: bytes, _mime_type: str, _context: str):
        raise TimeoutError("not sent to report")

    report = evaluate_multimodal_analyzer(
        [
            {
                "case_id": "geometry_01",
                "media": b"geometry-bytes",
                "mime_type": "image/png",
                "task_context": "识别辅助线关系",
                "expected_modality": "geometry",
                "expected_claim_terms": ["垂直"],
                "must_abstain": True,
            }
        ],
        unavailable,
    )

    assert report["cases"][0]["analyzer_error"] == "TimeoutError"
    assert report["metrics"]["modality_accuracy"] == 0.0
    assert report["metrics"]["grading_boundary_rate"] == 0.0


def _load_fixture(name: str) -> dict:
    return json.loads((ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8"))


def _offline_lockbox_analyzer(media: bytes, mime_type: str, context: str) -> dict:
    del mime_type, context
    tag = media.split(b"LOCKBOX:", 1)[1].decode()
    projection = {
        "scanned_pdf": ("scanned_pdf_page", "handwriting", "手写 公式", "abstain"),
        "table": ("spreadsheet", "table", "条件 结果", "observation"),
        "chart": ("image", "chart", "强光 8", "abstain"),
        "geometry": ("image", "geometry", "直角 AB", "requires_confirmation"),
        "handwriting": ("image", "handwriting", "字符 x", "abstain"),
        "formula": ("image", "formula", "f(x) x²", "requires_confirmation"),
        "slide_notes_conflict": (
            "slide",
            "mixed",
            "阈值 8 6",
            "requires_confirmation",
        ),
        "audio": ("audio", "unknown", "定义 状态", "abstain"),
        "video": ("video", "unknown", "状态 转移", "abstain"),
    }
    source, modality, observed, decision = projection[tag]
    temporal = source in {"audio", "video"}
    conflict = tag in {"geometry", "formula", "slide_notes_conflict"}
    remote = tag == "chart"
    claims = [
        {
            "claim_id": "multimodal_claim_01",
            "statement": observed,
            "evidence_locator": f"{source}/fixture-locator",
            "confidence": 0.9,
            "confidence_semantics": (
                "provider_observation_likelihood_not_correctness_or_mastery"
            ),
            "verification_status": "unverified_provider_observation",
            "untrusted_instruction_data": True,
            "grading_evidence_allowed": False,
            "mastery_evidence_allowed": False,
        }
    ]
    segments = (
        [
            {
                "segment_id": "segment_0001",
                "start_ms": 0,
                "end_ms": 1_000,
                "text": observed,
                "evidence_locator": f"{source}/0-1000ms",
                "confidence": 0.9,
                "grading_evidence_allowed": False,
                "mastery_evidence_allowed": False,
            }
        ]
        if temporal
        else []
    )
    conflicts = (
        [
            {
                "conflict_id": "multimodal_conflict_01",
                "kind": "cross_layer_disagreement",
                "description": "fixture conflict",
                "evidence_locators": ["layer/a", "layer/b"],
                "resolution_status": "unresolved",
            }
        ]
        if conflict
        else []
    )
    return {
        "schema": "teaching_skill_miner.multimodal_evidence.v2",
        "source_modality": source,
        "media_sha256": hashlib.sha256(media).hexdigest(),
        "provider_spec_sha256": "a" * 64,
        "provider_result_sha256": "b" * 64,
        "modality": modality,
        "transcription": {
            "status": "candidate",
            "text": observed,
            "confidence": 0.9,
            "segments": segments,
            "grading_evidence_allowed": False,
            "mastery_evidence_allowed": False,
        },
        "semantic_analysis": {
            "description": observed,
            "claims": claims,
            "uncertainties": ["fixture uncertainty"] if decision == "abstain" else [],
            "conflicts": conflicts,
            "grading_evidence_allowed": False,
            "mastery_evidence_allowed": False,
        },
        "claims": claims,
        "conflicts": conflicts,
        "decision": decision,
        "transcription_is_semantic_understanding": False,
        "semantic_analysis_is_answer_correctness": False,
        "grading_evidence_allowed": False,
        "mastery_evidence_allowed": False,
        "remote_media_sent": remote,
        "privacy_receipt": {
            "execution_scope": "remote" if remote else "local",
            "raw_media_sent": remote,
            "processing_region": "provider_managed" if remote else "on_device",
            "provider_retention_days": 0,
            "consent_id": "consent_fixture" if remote else None,
            "consent_receipt_sha256": "c" * 64 if remote else None,
            "consent_policy_version": 1 if remote else None,
            "data_categories": ["learner_image"] if remote else [],
        },
    }


def test_offline_maturity_lockbox_covers_every_required_modality_deterministically() -> None:
    inputs = _load_fixture("teacher_agent_multimodal_lockbox_inputs_v1.json")
    gold = _load_fixture("teacher_agent_multimodal_lockbox_gold_v1.json")
    received: list[tuple[bytes, str, str]] = []

    def analyzer(media: bytes, mime_type: str, context: str) -> dict:
        received.append((media, mime_type, context))
        return _offline_lockbox_analyzer(media, mime_type, context)

    first = evaluate_multimodal_lockbox(inputs, gold, analyzer)
    second = evaluate_multimodal_lockbox(inputs, gold, _offline_lockbox_analyzer)

    assert first == second
    assert len(received) == 9
    assert all(len(item) == 3 for item in received)
    assert first["coverage"] == [
        "audio",
        "chart",
        "formula",
        "geometry",
        "handwriting",
        "scanned_pdf",
        "slide_notes_conflict",
        "table",
        "video",
    ]
    assert set(first["metrics"].values()) == {1.0}
    assert first["claim_boundary"]["gold_hidden_from_analyzer"] is True
    assert first["claim_boundary"]["network_required"] is False
    assert first["claim_boundary"]["student_learning_effectiveness_established"] is False
    assert len(first["report_sha256"]) == 64


def test_lockbox_rejects_input_gold_tampering_and_incomplete_coverage() -> None:
    inputs = _load_fixture("teacher_agent_multimodal_lockbox_inputs_v1.json")
    gold = _load_fixture("teacher_agent_multimodal_lockbox_gold_v1.json")

    tampered_inputs = deepcopy(inputs)
    tampered_inputs["cases"][0]["task_context"] = "changed after gold sealing"
    with pytest.raises(MultimodalEvalError, match="binding"):
        evaluate_multimodal_lockbox(
            tampered_inputs, gold, _offline_lockbox_analyzer
        )

    incomplete_inputs = deepcopy(inputs)
    incomplete_gold = deepcopy(gold)
    incomplete_inputs["cases"].pop()
    incomplete_gold["cases"].pop()
    canonical = json.dumps(
        incomplete_inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    incomplete_gold["inputs_sha256"] = hashlib.sha256(canonical).hexdigest()
    with pytest.raises(MultimodalEvalError, match="coverage is incomplete"):
        evaluate_multimodal_lockbox(
            incomplete_inputs, incomplete_gold, _offline_lockbox_analyzer
        )


def test_lockbox_detects_raw_media_or_base64_leak_without_rejecting_hash_provenance() -> None:
    inputs = _load_fixture("teacher_agent_multimodal_lockbox_inputs_v1.json")
    gold = _load_fixture("teacher_agent_multimodal_lockbox_gold_v1.json")

    def leaking(media: bytes, mime_type: str, context: str) -> dict:
        evidence = _offline_lockbox_analyzer(media, mime_type, context)
        evidence["debug_raw_media"] = media.decode("utf-8", errors="replace")
        return evidence

    report = evaluate_multimodal_lockbox(inputs, gold, leaking)
    assert report["cases"][0]["raw_media_not_exposed"] is False
    assert report["metrics"]["raw_media_not_exposed"] < 1.0
    assert report["cases"][0]["provider_and_locator_provenance_held"] is True
