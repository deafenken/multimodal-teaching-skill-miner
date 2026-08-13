from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import jsonschema
import pytest

from teaching_skill_miner.io_utils import project_root, read_json
from teaching_skill_miner.teacher_agent import (
    _refresh_integrity,
    advance_teacher_agent_session,
    start_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_semantic_lockbox import (
    INPUT_SCHEMA,
    LEDGER_SCHEMA,
    SemanticLockboxError,
    build_semantic_runtime_ledger,
    score_semantic_lockbox,
    semantic_artifact_sha256,
    sign_semantic_lockbox_gold,
    sign_semantic_runtime_ledger,
    validate_semantic_lockbox_inputs,
    validate_semantic_runtime_ledger,
)


DOMAINS = (
    "mathematics",
    "computer_science",
    "physics",
    "chemistry",
    "biology",
    "history",
    "language_arts",
    "economics",
    "geography",
    "civics",
)


def _dataset() -> dict:
    cases = []
    for domain_index, domain in enumerate(DOMAINS):
        for kind_index, kind in enumerate(
            ("correct_paraphrase", "incorrect_plausible", "confusion")
        ):
            cases.append(
                {
                    "case_id": f"case_{domain_index:02d}_{kind_index:02d}",
                    "domain": domain,
                    "learner_text": f"{domain} learner response {kind}",
                    "response_kind": kind,
                    "authority_condition": (
                        "teacher_rubric" if kind != "confusion" else "no_authority"
                    ),
                    "event_round": 1,
                }
            )
    return {
        "schema": INPUT_SCHEMA,
        "benchmark_version": "1.0",
        "benchmark_id": "semantic_external_lockbox_001",
        "split": "external_held_out_lockbox",
        "gold_available_to_executor": False,
        "cases": cases,
        "claim_boundary": {
            "expert_labels_external": True,
            "real_students_involved": False,
            "deployment_accuracy_established": False,
            "real_learning_effect_established": False,
        },
    }


def _gold(dataset: dict) -> dict:
    rows = []
    for case in dataset["cases"]:
        correct = case["response_kind"] == "correct_paraphrase"
        confused = case["response_kind"] == "confusion"
        rows.append(
            {
                "case_id": case["case_id"],
                "expert_label": (
                    "correct"
                    if correct
                    else "confused"
                    if confused
                    else "misconception"
                ),
                "mastery_gain_allowed": correct,
                "phase_advance_allowed": correct,
                "abstention_required": not correct,
                "expert_review": {
                    "independent_reviewer_count": 2,
                    "adjudicated": True,
                    "agreement": "agreed",
                    "review_receipt_sha256": semantic_artifact_sha256(
                        {"case_id": case["case_id"], "review": "external"}
                    ),
                },
            }
        )
    return {
        "schema": "teaching_skill_miner.teacher_agent_semantic_lockbox_gold.v1",
        "benchmark_version": "1.0",
        "benchmark_id": dataset["benchmark_id"],
        "input_fingerprint": semantic_artifact_sha256(dataset),
        "visibility": "external_custodian_sealed",
        "cases": rows,
    }


def _keys() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _sessions(dataset: dict, *, corrupt_case_id: str | None = None) -> dict:
    root = project_root()
    demo = read_json(root / "data/teacher_agent_demo_input.json")
    library = read_json(root / "data/teacher_agent_skill_library.json")
    sessions = {}
    for case in dataset["cases"]:
        correct = case["response_kind"] == "correct_paraphrase"
        forced_false_positive = case["case_id"] == corrupt_case_id
        signal = "correct" if correct or forced_false_positive else "no_response"
        confidence = 0.95 if signal == "correct" else 0.0
        session = start_teacher_agent_session(
            demo["goal"], demo["student_profile"], library
        )
        session = advance_teacher_agent_session(
            session,
            learner_response=case["learner_text"],
            signal=signal,
            signal_confidence=confidence,
            answer_alignment="aligned" if signal == "correct" else "ambiguous",
            needs_human_review=signal != "correct",
            count_as_no_progress=False,
        )
        event = session["history"][-1]
        event["deepseek_assessment"] = {
            "semantic_entailment_established": correct,
            "teacher_grading_authority_available": correct,
            "needs_human_review": signal != "correct",
            "evidence_binding_source": (
                "teacher_knowledge_spec_exact_match"
                if correct
                else "model_excerpt_current_response_substring"
            ),
        }
        event["structured_signal"]["assessment_eligible"] = signal == "correct"
        event["structured_signal"]["applied_to_mastery"] = signal == "correct"
        sessions[case["case_id"]] = _refresh_integrity(session)
    return sessions


def _attestation(dataset: dict, gold: dict, private_pem: bytes, registration: str):
    return sign_semantic_lockbox_gold(
        gold,
        dataset,
        private_key_pem=private_pem,
        issuer="district_research_board",
        key_id="external_key_001",
        registration_id=registration,
        signed_at_utc="2026-08-12T00:00:00Z",
    )


def _runtime_attestation(dataset: dict, ledger: dict, private_pem: bytes):
    return sign_semantic_runtime_ledger(
        ledger,
        dataset,
        private_key_pem=private_pem,
        issuer="independent_runtime_lab",
        key_id="runtime_key_001",
        signed_at_utc="2026-08-12T00:00:00Z",
    )


def test_semantic_lockbox_scores_only_runtime_derived_externally_signed_evidence(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    gold = _gold(dataset)
    private_pem, public_pem = _keys()
    runtime_private_pem, runtime_public_pem = _keys()
    ledger = build_semantic_runtime_ledger(dataset, _sessions(dataset))
    report = score_semantic_lockbox(
        dataset,
        gold,
        ledger,
        _attestation(dataset, gold, private_pem, "registration_pass_001"),
        _runtime_attestation(dataset, ledger, runtime_private_pem),
        trusted_gold_public_key_pem=public_pem,
        trusted_runtime_public_key_pem=runtime_public_pem,
        consumption_ledger_dir=tmp_path / "consumed",
    )

    assert ledger["schema"] == LEDGER_SCHEMA
    assert "learner response" not in json.dumps(ledger)
    assert report["passed"] is True
    assert report["metrics"] == {
        "semantic_false_positive_rate": 0.0,
        "false_mastery_rate": 0.0,
        "unauthorized_mastery_rate": 0.0,
        "phase_advance_violation_rate": 0.0,
        "correct_paraphrase_recall": 1.0,
        "required_abstention_rate": 1.0,
        "brier_score": pytest.approx(0.000833, abs=1e-6),
        "expected_calibration_error": pytest.approx(0.016667, abs=1e-6),
    }
    assert report["claim_boundary"] == {
        "expert_gold_signature_verified": True,
        "gold_hidden_from_executor": True,
        "runtime_observations_derived_from_integrity_checked_sessions": True,
        "behavioral_thresholds_met": True,
        "deployment_accuracy_established": False,
        "real_students_involved": False,
        "real_learning_effect_established": False,
    }
    assert report["private_case_diagnostics_included"] is False
    assert "case_reports" not in report
    consumption = json.loads(
        (tmp_path / "consumed" / "registration_pass_001.json").read_text()
    )
    assert consumption["status"] == "completed"
    assert (
        consumption["evaluation_material_sha256"]
        == report["provenance"]["evaluation_material_sha256"]
    )


def test_semantic_lockbox_exposes_quoted_text_false_mastery_attack(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    gold = _gold(dataset)
    attacked_case = next(
        case
        for case in dataset["cases"]
        if case["response_kind"] == "incorrect_plausible"
    )
    private_pem, public_pem = _keys()
    runtime_private_pem, runtime_public_pem = _keys()
    ledger = build_semantic_runtime_ledger(
        dataset, _sessions(dataset, corrupt_case_id=attacked_case["case_id"])
    )
    report = score_semantic_lockbox(
        dataset,
        gold,
        ledger,
        _attestation(dataset, gold, private_pem, "registration_attack_001"),
        _runtime_attestation(dataset, ledger, runtime_private_pem),
        trusted_gold_public_key_pem=public_pem,
        trusted_runtime_public_key_pem=runtime_public_pem,
        consumption_ledger_dir=tmp_path / "consumed",
    )

    assert report["passed"] is False
    assert report["metrics"]["semantic_false_positive_rate"] > 0.01
    assert report["metrics"]["false_mastery_rate"] > 0.01
    assert report["metrics"]["unauthorized_mastery_rate"] > 0.0


def test_semantic_lockbox_rejects_gold_tampering_and_reuse(tmp_path: Path) -> None:
    dataset = _dataset()
    gold = _gold(dataset)
    private_pem, public_pem = _keys()
    runtime_private_pem, runtime_public_pem = _keys()
    attestation = _attestation(dataset, gold, private_pem, "registration_once_001")
    ledger = build_semantic_runtime_ledger(dataset, _sessions(dataset))
    score_semantic_lockbox(
        dataset,
        gold,
        ledger,
        attestation,
        _runtime_attestation(dataset, ledger, runtime_private_pem),
        trusted_gold_public_key_pem=public_pem,
        trusted_runtime_public_key_pem=runtime_public_pem,
        consumption_ledger_dir=tmp_path / "consumed",
    )
    with pytest.raises(SemanticLockboxError, match="already consumed"):
        score_semantic_lockbox(
            dataset,
            gold,
            ledger,
            attestation,
            _runtime_attestation(dataset, ledger, runtime_private_pem),
            trusted_gold_public_key_pem=public_pem,
            trusted_runtime_public_key_pem=runtime_public_pem,
            consumption_ledger_dir=tmp_path / "consumed",
        )

    tampered = deepcopy(gold)
    tampered["cases"][0]["expert_label"] = "misconception"
    with pytest.raises(SemanticLockboxError, match="attestation binding"):
        score_semantic_lockbox(
            dataset,
            tampered,
            ledger,
            attestation,
            _runtime_attestation(dataset, ledger, runtime_private_pem),
            trusted_gold_public_key_pem=public_pem,
            trusted_runtime_public_key_pem=runtime_public_pem,
            consumption_ledger_dir=tmp_path / "other",
        )


def test_semantic_lockbox_rejects_self_report_and_weak_domain_coverage(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    weak = deepcopy(dataset)
    weak["cases"] = weak["cases"][:27]
    with pytest.raises(SemanticLockboxError, match="at least 30"):
        validate_semantic_lockbox_inputs(weak)

    ledger = build_semantic_runtime_ledger(dataset, _sessions(dataset))
    forged = deepcopy(ledger)
    negative = next(
        row for row in forged["cases"] if row["observed_signal"] != "correct"
    )
    negative["observed_signal"] = "correct"
    with pytest.raises(SemanticLockboxError, match="content hash"):
        validate_semantic_runtime_ledger(forged, dataset)

    # Recomputing a plain content hash still must not turn a prediction into a
    # trusted runtime receipt.  Only the separately trusted runner signature
    # can authenticate the exact ledger.
    material = deepcopy(forged)
    material.pop("content_sha256")
    forged["content_sha256"] = semantic_artifact_sha256(material)
    validate_semantic_runtime_ledger(forged, dataset)
    gold = _gold(dataset)
    gold_private, gold_public = _keys()
    runtime_private, runtime_public = _keys()
    original = build_semantic_runtime_ledger(dataset, _sessions(dataset))
    with pytest.raises(SemanticLockboxError, match="runtime attestation binding"):
        score_semantic_lockbox(
            dataset,
            gold,
            forged,
            _attestation(dataset, gold, gold_private, "registration_forged_001"),
            _runtime_attestation(dataset, original, runtime_private),
            trusted_gold_public_key_pem=gold_public,
            trusted_runtime_public_key_pem=runtime_public,
            consumption_ledger_dir=tmp_path / "consumed",
        )


def test_semantic_lockbox_requires_independent_gold_and_runtime_keys(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    gold = _gold(dataset)
    ledger = build_semantic_runtime_ledger(dataset, _sessions(dataset))
    private_pem, public_pem = _keys()
    with pytest.raises(SemanticLockboxError, match="must be independent"):
        score_semantic_lockbox(
            dataset,
            gold,
            ledger,
            _attestation(dataset, gold, private_pem, "registration_shared_001"),
            _runtime_attestation(dataset, ledger, private_pem),
            trusted_gold_public_key_pem=public_pem,
            trusted_runtime_public_key_pem=public_pem,
            consumption_ledger_dir=tmp_path / "consumed",
        )


def test_semantic_lockbox_public_schemas_accept_generated_artifacts(
    tmp_path: Path,
) -> None:
    root = project_root()
    dataset = _dataset()
    gold = _gold(dataset)
    ledger = build_semantic_runtime_ledger(dataset, _sessions(dataset))
    gold_private, gold_public = _keys()
    runtime_private, runtime_public = _keys()
    gold_attestation = _attestation(
        dataset, gold, gold_private, "registration_schema_001"
    )
    runtime_attestation = _runtime_attestation(dataset, ledger, runtime_private)
    report = score_semantic_lockbox(
        dataset,
        gold,
        ledger,
        gold_attestation,
        runtime_attestation,
        trusted_gold_public_key_pem=gold_public,
        trusted_runtime_public_key_pem=runtime_public,
        consumption_ledger_dir=tmp_path / "consumed",
    )
    for artifact, schema_name in (
        (dataset, "teacher_agent_semantic_lockbox_inputs.schema.json"),
        (gold, "teacher_agent_semantic_lockbox_gold.schema.json"),
        (ledger, "teacher_agent_semantic_runtime_ledger.schema.json"),
        (gold_attestation, "teacher_agent_semantic_lockbox_attestation.schema.json"),
        (runtime_attestation, "teacher_agent_semantic_lockbox_attestation.schema.json"),
        (report, "teacher_agent_semantic_lockbox_report.schema.json"),
    ):
        schema = json.loads((root / "schema" / schema_name).read_text())
        jsonschema.Draft202012Validator(schema).validate(artifact)
