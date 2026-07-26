"""End-to-end delivery verification with explicit external-evidence boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import mean
from typing import Any

from .attestation import verify_evaluation_report_receipt
from .audit import audit_dataset
from .benchmark import benchmark_transfer
from .evaluator import evaluate_collection, evaluate_skill
from .external_evidence import verify_external_research_evidence_files
from .human_eval import skill_review_fingerprint, summarize_human_review
from .io_utils import read_json, resolve_manifest_root
from .miner import mine_skill
from .models import validate_skill, validate_transcript
from .recognition.strict_evaluation import frozen_evaluation_report_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human_incomplete(expected_skill_ids: list[str], reason: str) -> dict[str, Any]:
    return {
        "validation_status": "incomplete",
        "expected_skill_count": len(expected_skill_ids),
        "reviewed_skill_count": 0,
        "missing_skill_ids": expected_skill_ids,
        "extra_skill_ids": [],
        "passed": False,
        "errors": [reason],
        "skills": [],
    }


def _dipser_status(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "integrity_metadata_present": False,
            "claim_boundaries_safe": None,
            "note": "No DIPSER report was supplied; recognition appendix verification was skipped.",
        }
    report = read_json(path)
    binding = report.get("artifact_binding", {}) if isinstance(report, dict) else {}
    claims = report.get("claim_status", {}) if isinstance(report, dict) else {}
    expected_false = (
        "accuracy_0_9_established",
        "deployment_accuracy_established",
        "multimodal_gain_established",
        "real_time_accuracy_established",
    )
    claim_boundaries_safe = all(claims.get(name) is False for name in expected_false)
    causal = report.get("strict_causal_reference", {}) if isinstance(report, dict) else {}
    causal_metrics = causal.get("session_sgkf5", {}) if isinstance(causal, dict) else {}
    nested = report.get("nested_offline_full_session", {}) if isinstance(report, dict) else {}
    nested_sgkf = nested.get("sgkf5", {}) if isinstance(nested, dict) else {}
    nested_loso = nested.get("leave_one_session_out", {}) if isinstance(nested, dict) else {}
    nested_metrics = (
        nested_sgkf.get("nested_hierarchical_metrics", {})
        if isinstance(nested_sgkf, dict)
        else {}
    )
    loso_metrics = (
        nested_loso.get("nested_hierarchical_metrics", {})
        if isinstance(nested_loso, dict)
        else {}
    )
    return {
        "available": True,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "integrity_metadata_present": bool(
            binding.get("dataset_fingerprint")
            and binding.get("feature_bundle_fingerprint")
            and binding.get("sample_order_verified") is True
        ),
        "claim_boundaries_safe": claim_boundaries_safe,
        "claim_status": claims,
        "strict_causal_accuracy": causal_metrics.get("accuracy"),
        "strict_causal_macro_f1": causal_metrics.get("macro_f1"),
        "offline_full_session_accuracy": nested_metrics.get("accuracy"),
        "offline_full_session_macro_f1": nested_metrics.get("macro_f1"),
        "offline_full_session_loso_accuracy": loso_metrics.get("accuracy"),
        "offline_full_session_loso_macro_f1": loso_metrics.get("macro_f1"),
        "note": "The 0.9 candidate remains an offline post-selection development estimate, not deployment accuracy.",
    }


def _external_deployment_status(
    report_path: Path | None,
    receipt_path: Path | None,
    trusted_public_key_path: Path | None,
) -> dict[str, Any]:
    supplied = [report_path is not None, receipt_path is not None, trusted_public_key_path is not None]
    if not any(supplied):
        return {
            "available": False,
            "passed": False,
            "reason": "No signed prospective deployment report was supplied.",
        }
    if not all(supplied):
        return {
            "available": True,
            "passed": False,
            "reason": "Deployment report, signed receipt, and trusted public key must be supplied together.",
        }
    assert report_path is not None and receipt_path is not None and trusted_public_key_path is not None
    if not report_path.is_file() or not receipt_path.is_file() or not trusted_public_key_path.is_file():
        return {
            "available": True,
            "passed": False,
            "reason": "One or more signed deployment evidence files do not exist.",
        }
    try:
        report = read_json(report_path)
        receipt = read_json(receipt_path)
        if not isinstance(report, dict) or not isinstance(receipt, dict):
            raise ValueError("deployment evidence must be JSON objects")
        expected_report_fingerprint = frozen_evaluation_report_fingerprint(report)
        if report.get("evaluation_report_fingerprint") != expected_report_fingerprint:
            raise ValueError("deployment report fingerprint mismatch")
        registration = report.get("freeze_registration")
        if not isinstance(registration, dict) or registration.get("verified") is not True:
            raise ValueError("deployment report lacks a verified freeze registration")
        verification = verify_evaluation_report_receipt(
            receipt,
            trusted_public_key_pem=trusted_public_key_path.read_bytes(),
            expected_report_fingerprint=expected_report_fingerprint,
            expected_registration_id=str(registration.get("registration_id", "")),
            expected_registration_attestation_fingerprint=str(
                registration.get("attestation_fingerprint", "")
            ),
        )
        if verification["public_key_sha256"] != registration.get("public_key_sha256"):
            raise ValueError("registration and final receipt use different custodian keys")
        one_time_receipt = report.get("one_time_consumption_receipt")
        one_time_complete = bool(
            isinstance(one_time_receipt, dict)
            and one_time_receipt.get("status") == "completed"
            and one_time_receipt.get("evaluation_fingerprint")
            == expected_report_fingerprint
        )
        passed = bool(
            report.get("protocol")
            == "frozen_external_deployment_evaluation_v3_signed_registration"
            and report.get("deployment_accuracy_established") is True
            and report.get("all_frozen_claim_gates_passed") is True
            and report.get("prospective_deployment_collection") is True
            and report.get("one_time_lockbox_evaluation") is True
            and one_time_complete
            and verification.get("deployment_accuracy_established") is True
        )
        return {
            "available": True,
            "passed": passed,
            "report_path": str(report_path),
            "report_sha256": _sha256(report_path),
            "receipt_path": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "trusted_public_key_sha256": verification["public_key_sha256"],
            "evaluation_report_fingerprint": expected_report_fingerprint,
            "registration_id": registration.get("registration_id"),
            "metrics": report.get("metrics"),
            "reason": (
                "Signed prospective deployment evidence passed all frozen gates."
                if passed
                else "Signed files were valid, but one or more deployment gates failed."
            ),
        }
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            "available": True,
            "passed": False,
            "reason": f"Signed deployment evidence verification failed: {exc}",
        }


def _external_research_status(
    evidence_kind: str,
    manifest_path: Path | None,
    attestation_path: Path | None,
    trusted_public_key_path: Path | None,
    evaluated_system_artifact_path: Path | None,
) -> dict[str, Any]:
    supplied = [
        manifest_path is not None,
        attestation_path is not None,
        trusted_public_key_path is not None,
    ]
    claim_label = {
        "confirmatory_multimodal_gain": "confirmatory multimodal-gain",
        "real_learner_effectiveness": "real learner-effectiveness",
    }[evidence_kind]
    if not any(supplied):
        return {
            "available": False,
            "passed": False,
            "evidence_kind": evidence_kind,
            "reason": f"No signed external {claim_label} evidence was supplied.",
        }
    if not all(supplied):
        return {
            "available": True,
            "passed": False,
            "evidence_kind": evidence_kind,
            "reason": (
                f"External {claim_label} manifest, attestation, and its own "
                "out-of-band trusted public key must be supplied together."
            ),
        }
    if evaluated_system_artifact_path is None:
        return {
            "available": True,
            "passed": False,
            "evidence_kind": evidence_kind,
            "reason": (
                f"External {claim_label} evidence was supplied without the exact "
                "local evaluated-system artifact needed for SHA-256 binding."
            ),
        }
    assert (
        manifest_path is not None
        and attestation_path is not None
        and trusted_public_key_path is not None
    )
    try:
        verification = verify_external_research_evidence_files(
            manifest_path,
            attestation_path,
            trusted_public_key_path,
            expected_kind=evidence_kind,
            expected_system_artifact_path=evaluated_system_artifact_path,
        )
        passed = bool(
            verification.get("signature_verified") is True
            and verification.get("claim_established") is True
            and verification.get("system_artifact_binding_verified") is True
        )
        return {
            "available": True,
            "passed": passed,
            **verification,
            "reason": (
                f"Signed external {claim_label} evidence passed every recomputed gate."
                if passed
                else (
                    f"Signed external {claim_label} evidence was authentic, but the "
                    "recomputed claim gates did not all pass."
                )
            ),
        }
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            "available": True,
            "passed": False,
            "evidence_kind": evidence_kind,
            "reason": f"Signed external {claim_label} evidence verification failed: {exc}",
        }


def verify_delivery(
    manifest_path: str | Path,
    cases_path: str | Path,
    *,
    formal_manifest_path: str | Path | None = None,
    human_review_path: str | Path | None = None,
    dipser_report_path: str | Path | None = None,
    external_deployment_report_path: str | Path | None = None,
    external_evaluation_receipt_path: str | Path | None = None,
    trusted_attestation_public_key_path: str | Path | None = None,
    external_multimodal_evidence_path: str | Path | None = None,
    external_multimodal_attestation_path: str | Path | None = None,
    trusted_multimodal_public_key_path: str | Path | None = None,
    external_learner_evidence_path: str | Path | None = None,
    external_learner_attestation_path: str | Path | None = None,
    trusted_learner_public_key_path: str | Path | None = None,
    external_evaluated_system_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the core evidence in memory and report what is and is not complete."""

    manifest_file = Path(manifest_path).resolve()
    cases_file = Path(cases_path).resolve()
    manifest = read_json(manifest_file)
    root = resolve_manifest_root(manifest_file, manifest)
    dataset_report = audit_dataset(manifest, root)
    formal_manifest_file = (
        Path(formal_manifest_path).resolve()
        if formal_manifest_path is not None
        else manifest_file
    )
    if formal_manifest_path is not None:
        formal_manifest = read_json(formal_manifest_file)
        formal_root = resolve_manifest_root(formal_manifest_file, formal_manifest)
        formal_dataset_report = audit_dataset(formal_manifest, formal_root)
    else:
        formal_dataset_report = dataset_report
    skills: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    transcript_rows: list[dict[str, Any]] = []
    for item in manifest.get("videos", []):
        transcript_path = root / str(item.get("transcript_path", ""))
        transcript = read_json(transcript_path)
        transcript_validation = validate_transcript(transcript)
        skill = mine_skill(transcript)
        skill_validation = validate_skill(skill)
        evaluation = evaluate_skill(skill, transcript)
        transcript_rows.append(
            {
                "video_id": item.get("video_id"),
                "transcript_path": str(transcript_path),
                "transcript_valid": transcript_validation.valid,
                "skill_id": skill.get("skill_id"),
                "skill_valid": skill_validation.valid,
                "evaluation_passed": evaluation.get("passed"),
                "evaluation_score": evaluation.get("overall_score"),
                "procedure_evidence_audit": evaluation.get("method_provenance"),
            }
        )
        skills.append(skill)
        evaluations.append(evaluation)
    collection = evaluate_collection(skills, evaluations, manifest)
    cases_payload = read_json(cases_file)
    cases = cases_payload.get("cases", cases_payload) if isinstance(cases_payload, dict) else cases_payload
    transfer = benchmark_transfer(skills, cases)
    expected_skill_ids = sorted(str(skill["skill_id"]) for skill in skills)
    expected_skill_fingerprints = {
        str(skill["skill_id"]): skill_review_fingerprint(skill) for skill in skills
    }
    review_path = Path(human_review_path).resolve() if human_review_path else None
    if review_path and review_path.is_file():
        human = summarize_human_review(
            review_path,
            expected_skill_ids=expected_skill_ids,
            expected_skill_fingerprints=expected_skill_fingerprints,
        )
        human["path"] = str(review_path)
    else:
        human = _human_incomplete(expected_skill_ids, "Human review CSV is missing or was not supplied.")
    dipser_path = Path(dipser_report_path).resolve() if dipser_report_path else None
    dipser = _dipser_status(dipser_path)
    deployment = _external_deployment_status(
        Path(external_deployment_report_path).resolve()
        if external_deployment_report_path
        else None,
        Path(external_evaluation_receipt_path).resolve()
        if external_evaluation_receipt_path
        else None,
        Path(trusted_attestation_public_key_path).resolve()
        if trusted_attestation_public_key_path
        else None,
    )
    multimodal_evidence = _external_research_status(
        "confirmatory_multimodal_gain",
        Path(external_multimodal_evidence_path).resolve()
        if external_multimodal_evidence_path
        else None,
        Path(external_multimodal_attestation_path).resolve()
        if external_multimodal_attestation_path
        else None,
        Path(trusted_multimodal_public_key_path).resolve()
        if trusted_multimodal_public_key_path
        else None,
        Path(external_evaluated_system_artifact_path).resolve()
        if external_evaluated_system_artifact_path
        else None,
    )
    learner_evidence = _external_research_status(
        "real_learner_effectiveness",
        Path(external_learner_evidence_path).resolve()
        if external_learner_evidence_path
        else None,
        Path(external_learner_attestation_path).resolve()
        if external_learner_attestation_path
        else None,
        Path(trusted_learner_public_key_path).resolve()
        if trusted_learner_public_key_path
        else None,
        Path(external_evaluated_system_artifact_path).resolve()
        if external_evaluated_system_artifact_path
        else None,
    )
    local_checks = {
        "dataset_structure": dataset_report["dataset_structure_passed"],
        "two_courses_five_lessons_each": dataset_report["structure_checks"]["at_least_two_courses"]
        and dataset_report["structure_checks"]["at_least_five_lessons_per_course"],
        "ten_valid_transcripts": len(transcript_rows) >= 10
        and all(row["transcript_valid"] for row in transcript_rows),
        "skill_for_every_video": len(skills) == len(manifest.get("videos", []))
        and all(row["skill_valid"] for row in transcript_rows),
        "all_automatic_skill_evaluations_pass": bool(evaluations)
        and all(value.get("passed") for value in evaluations),
        "collection_requirements_pass": collection["passed"],
        "transfer_benchmark_pass": transfer["passed"],
        "recognition_claim_boundaries_safe": not dipser["available"]
        or (
            dipser["integrity_metadata_present"]
            and dipser["claim_boundaries_safe"] is True
        ),
    }
    external_checks = {
        "formal_full_transcripts_or_asr": formal_dataset_report[
            "formal_empirical_ready"
        ],
        "independent_human_review_complete": human.get("passed") is True,
        "confirmatory_external_multimodal_gain": multimodal_evidence["passed"],
        "prospective_deployment_accuracy": deployment["passed"],
        "real_learner_effectiveness_established": learner_evidence["passed"],
    }
    external_blockers = []
    if not external_checks["formal_full_transcripts_or_asr"]:
        external_blockers.append(
            "Supply a separate complete official-caption or independently audited-ASR manifest; the bundled curated paraphrase excerpts are demo-only."
        )
    if not external_checks["independent_human_review_complete"]:
        external_blockers.append(
            "Collect two independent human reviews for every expected Skill; the repository intentionally ships no fabricated scores."
        )
    if not external_checks["confirmatory_external_multimodal_gain"]:
        external_blockers.append(
            "Supply preregistered, identity-disjoint real-world paired modality-ablation evidence, signed by an externally governed trusted key."
        )
    if not external_checks["prospective_deployment_accuracy"]:
        external_blockers.append(
            "Run the frozen model once on a genuinely held-out target-site lockbox satisfying the preregistered claim contract."
        )
    if not external_checks["real_learner_effectiveness_established"]:
        external_blockers.append(
            "Supply a preregistered, ethically approved randomized learner study with a meaningful positive effect, signed by an externally governed trusted key."
        )
    engineering_ready = all(local_checks.values())
    validation_complete = all(external_checks.values())
    return {
        "schema_version": "1.0",
        "verification_kind": "teaching_skill_miner_delivery_verification",
        "manifest": str(manifest_file),
        "cases": str(cases_file),
        "engineering_delivery_ready": engineering_ready,
        "engineering_delivery_scope": (
            "bundled deterministic dataset/Skill/evaluation/transfer evidence only; "
            "does not verify tests, exact-wheel installation, clean builds, CI, tracked-file "
            "privacy, media-ASR readiness, or any external research claim"
        ),
        "research_validation_complete": validation_complete,
        "overall_status": (
            "complete" if engineering_ready and validation_complete else
            "engineering_ready_external_validation_pending" if engineering_ready else
            "engineering_incomplete"
        ),
        "local_checks": local_checks,
        "external_evidence_checks": external_checks,
        "external_blockers": external_blockers,
        "dataset": dataset_report,
        "formal_transcript_dataset": {
            "manifest": str(formal_manifest_file),
            "separate_from_demo_manifest": formal_manifest_file != manifest_file,
            "audit": formal_dataset_report,
        },
        "collection": collection,
        "transfer_benchmark": transfer,
        "human_validation": human,
        "recognition_appendix": dipser,
        "external_deployment_validation": deployment,
        "external_confirmatory_multimodal_validation": multimodal_evidence,
        "external_learner_effectiveness_validation": learner_evidence,
        "transcripts_and_skills": transcript_rows,
        "summary_metrics": {
            "video_count": len(transcript_rows),
            "skill_count": len(skills),
            "average_automatic_score": round(
                mean(float(value["overall_score"]) for value in evaluations), 2
            ) if evaluations else 0.0,
            "automatic_pass_rate": round(
                sum(bool(value.get("passed")) for value in evaluations) / len(evaluations), 4
            ) if evaluations else 0.0,
            "human_reviewed_skill_count": human.get("reviewed_skill_count", 0),
        },
        "claim_policy": {
            "automatic_scores_are_learning_effectiveness": False,
            "offline_transductive_accuracy_is_deployment_accuracy": False,
            "engineering_delivery_ready_means_exact_release_verified": False,
            "missing_external_evidence_may_be_synthesized": False,
            "signature_alone_proves_signer_independence": False,
            "trusted_external_keys_must_be_provisioned_out_of_band": True,
        },
    }


def delivery_markdown(report: dict[str, Any]) -> str:
    checks = report["local_checks"]
    external = report["external_evidence_checks"]
    lines = [
        "# 项目交付验收报告",
        "",
        f"- 内置核心工程证据：{'通过' if report['engineering_delivery_ready'] else '未通过'}",
        "- 范围：不替代测试、exact wheel 安装、干净构建、CI、tracked-file 隐私、真实 ASR 或外部研究验收",
        f"- 研究验证：{'完成' if report['research_validation_complete'] else '外部证据待完成'}",
        f"- 总状态：`{report['overall_status']}`",
        "",
        "## 本地可复现检查",
        "",
        "| 检查 | 结果 |",
        "|---|:---:|",
    ]
    for name, passed in checks.items():
        lines.append(f"| `{name}` | {'通过' if passed else '未通过'} |")
    lines.extend(["", "## 外部证据检查", "", "| 检查 | 结果 |", "|---|:---:|"])
    for name, passed in external.items():
        lines.append(f"| `{name}` | {'完成' if passed else '待完成'} |")
    lines.extend(["", "## 不能由项目伪造补齐的事项", ""])
    lines.extend(f"- {value}" for value in report["external_blockers"])
    lines.extend(
        [
            "",
            "自动评分只表示结构、证据一致性、可执行性与迁移能力覆盖；不等价于真实学习效果。",
            "离线完整 session 的 0.9 候选不等价于实时或部署准确率。",
            "",
        ]
    )
    return "\n".join(lines)
