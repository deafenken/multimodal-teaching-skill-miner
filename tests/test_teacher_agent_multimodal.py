from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_consent import RemoteConsentStore
from teaching_skill_miner.teacher_agent_multimodal import (
    MULTIMODAL_PROVIDER_RESULT_SCHEMA,
    MULTIMODAL_PROVIDER_SPEC_SCHEMA,
    VISUAL_PROVIDER_RESULT_SCHEMA,
    VisualSemanticError,
    analyze_multimodal_semantics,
    analyze_temporal_media,
    analyze_visual_semantics,
    multimodal_provider_spec,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"bounded-test-image"


class _Provider:
    def __init__(self, *, remote: bool) -> None:
        self.provider_id = "vision-test-v1"
        self.processing_region = "provider_managed" if remote else "on_device"
        self.sends_raw_media_remotely = remote
        self.calls = 0

    def analyze(self, image_bytes, mime_type, *, task_context):
        self.calls += 1
        assert image_bytes == PNG
        assert mime_type == "image/png"
        assert task_context == "解释图中两组柱形数据"
        return {
            "schema": VISUAL_PROVIDER_RESULT_SCHEMA,
            "modality": "chart",
            "description": "两组柱形对象，各有两个标注值。",
            "claims": [
                {
                    "statement": "左侧第一根柱标注为 2",
                    "evidence_locator": "left-group/bar-1/label",
                    "confidence": 0.91,
                }
            ],
            "uncertainties": ["纵轴单位不可见"],
        }


def test_local_visual_semantics_are_auditable_but_never_a_grading_key() -> None:
    provider = _Provider(remote=False)
    evidence = analyze_visual_semantics(
        PNG,
        "image/png",
        task_context="解释图中两组柱形数据",
        provider=provider,
    )

    assert provider.calls == 1
    assert evidence["remote_media_sent"] is False
    assert evidence["raw_media_retained"] is False
    assert evidence["modality"] == "chart"
    assert evidence["grading_evidence_allowed"] is False
    assert evidence["claims"][0]["grading_evidence_allowed"] is False
    assert evidence["provider_confidence_is_answer_correctness"] is False
    assert evidence["requires_teacher_review"] is True
    assert evidence["consent_receipt_sha256"] is None
    assert "bounded-test-image" not in json.dumps(evidence)


def test_remote_visual_provider_is_not_called_without_exact_server_receipt(
    tmp_path,
) -> None:
    provider = _Provider(remote=True)
    store = RemoteConsentStore(
        tmp_path / "consent.json",
        signing_secret=b"v" * 32,
        clock=lambda: datetime(2026, 8, 12, tzinfo=timezone.utc),
    )
    with pytest.raises(VisualSemanticError, match="requires a server consent"):
        analyze_visual_semantics(
            PNG,
            "image/png",
            task_context="解释图中两组柱形数据",
            provider=provider,
        )
    assert provider.calls == 0

    wrong = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="public_web_search",
        provider_id="vision-test-v1",
        processing_region="provider_managed",
        data_categories=["public_web_query"],
        provider_retention_days=0,
    )
    with pytest.raises(VisualSemanticError, match="consent is invalid"):
        analyze_visual_semantics(
            PNG,
            "image/png",
            task_context="解释图中两组柱形数据",
            provider=provider,
            subject_id="subject_abcdefghijklmnop",
            consent_id=wrong["consent_id"],
            consent_store=store,
        )
    assert provider.calls == 0

    receipt = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_visual_analysis",
        provider_id="vision-test-v1",
        processing_region="provider_managed",
        data_categories=["learner_image"],
        provider_retention_days=0,
    )
    evidence = analyze_visual_semantics(
        PNG,
        "image/png",
        task_context="解释图中两组柱形数据",
        provider=provider,
        subject_id="subject_abcdefghijklmnop",
        consent_id=receipt["consent_id"],
        consent_store=store,
    )
    assert provider.calls == 1
    assert evidence["remote_media_sent"] is True
    assert evidence["consent_receipt_sha256"] == receipt["receipt_sha256"]


def test_malformed_visual_result_fails_closed_without_persisting_raw_media() -> None:
    provider = _Provider(remote=False)

    def invalid(*_args, **_kwargs):
        return {
            "schema": VISUAL_PROVIDER_RESULT_SCHEMA,
            "modality": "chart",
            "description": "猜测",
            "claims": [
                {
                    "statement": "柱子更高",
                    "evidence_locator": "unknown",
                    "confidence": 2.0,
                }
            ],
            "uncertainties": [],
        }

    provider.analyze = invalid
    with pytest.raises(VisualSemanticError, match="claim 0"):
        analyze_visual_semantics(
            PNG,
            "image/png",
            task_context="解释图中两组柱形数据",
            provider=provider,
        )


class _V2Provider:
    def __init__(
        self,
        *,
        source_modalities: list[str],
        mime_types: list[str],
        result: dict,
        remote: bool = False,
        region: str = "on_device",
        retention_days: int | None = 0,
        capabilities: list[str] | None = None,
    ) -> None:
        self.provider_id = "stable-multimodal-test-v2"
        self.processing_region = region
        self.sends_raw_media_remotely = remote
        self.provider_spec = {
            "schema": MULTIMODAL_PROVIDER_SPEC_SCHEMA,
            "provider_id": self.provider_id,
            "adapter_version": "2.1.0",
            "execution_scope": "remote" if remote else "local",
            "processing_region": region,
            "raw_media_transport": "remote" if remote else "none",
            "provider_retention_days": retention_days,
            "supported_source_modalities": sorted(source_modalities),
            "supported_mime_types": sorted(mime_types),
            "capabilities": sorted(
                capabilities or ["semantic_analysis", "transcription"]
            ),
            "deterministic": True,
        }
        self.result = result
        self.calls = 0

    def analyze(self, media_bytes, mime_type, *, task_context):
        del media_bytes, mime_type, task_context
        self.calls += 1
        return self.result


def _v2_result(
    *,
    source: str = "image",
    modality: str = "geometry",
    decision: str = "observation",
    uncertainties: list[str] | None = None,
    conflicts: list[dict] | None = None,
) -> dict:
    return {
        "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
        "source_modality": source,
        "modality": modality,
        "transcription": {
            "status": "candidate",
            "text": "AB 垂直 CD",
            "confidence": 0.93,
            "language": "zh-Hans",
            "segments": [],
        },
        "semantic_analysis": {
            "status": "conflict" if conflicts else "provider_observation",
            "description": "图中标记显示 AB 与 CD 的关系。",
            "claims": [
                {
                    "statement": "AB 与 CD 之间可见直角标记",
                    "evidence_locator": "image/right-angle-marker-1",
                    "confidence": 0.87,
                }
            ],
            "uncertainties": uncertainties or [],
            "conflicts": conflicts or [],
        },
        "decision": decision,
    }


def test_v2_contract_separates_transcription_semantics_and_assessment() -> None:
    provider = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        result=_v2_result(),
    )
    evidence = analyze_multimodal_semantics(
        PNG,
        "image/png",
        task_context="描述几何图中的可见标记",
        provider=provider,
    )

    assert evidence["transcription"]["text"] == "AB 垂直 CD"
    assert evidence["transcription"]["grading_evidence_allowed"] is False
    assert evidence["transcription_is_semantic_understanding"] is False
    assert evidence["semantic_analysis_is_answer_correctness"] is False
    assert evidence["visual_verification_status"] == "not_verified"
    assert evidence["grading_evidence_allowed"] is False
    assert evidence["mastery_evidence_allowed"] is False
    assert evidence["claims"][0]["verification_status"] == (
        "unverified_provider_observation"
    )
    assert evidence["privacy_receipt"] == {
        "execution_scope": "local",
        "raw_media_sent": False,
        "processing_region": "on_device",
        "provider_retention_days": 0,
        "consent_id": None,
        "consent_receipt_sha256": None,
        "consent_policy_version": None,
        "data_categories": [],
    }
    canonical_spec = multimodal_provider_spec(provider)
    assert canonical_spec["compatibility_adapter"] is False
    provider.provider_spec = canonical_spec
    replay = analyze_multimodal_semantics(
        PNG,
        "image/png",
        task_context="描述几何图中的可见标记",
        provider=provider,
    )
    assert replay["provider"] == canonical_spec


def test_conflict_requires_confirmation_and_uncertainty_cannot_be_observation() -> None:
    conflict = {
        "kind": "ambiguous_visual_relation",
        "description": "直角标记可能属于相邻线段。",
        "evidence_locators": ["image/marker", "image/adjacent-segment"],
    }
    provider = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        result=_v2_result(
            decision="requires_confirmation", conflicts=[conflict]
        ),
    )
    evidence = analyze_multimodal_semantics(
        PNG,
        "image/png",
        task_context="描述几何图中的可见标记",
        provider=provider,
    )
    assert evidence["decision"] == "requires_confirmation"
    assert evidence["requires_learner_confirmation"] is True
    assert evidence["conflicts"][0]["resolution_status"] == "unresolved"
    assert evidence["mastery_evidence_allowed"] is False

    provider.result = _v2_result(
        decision="observation", uncertainties=["手写符号无法区分 1 与 l"]
    )
    with pytest.raises(VisualSemanticError, match="uncertainty requires"):
        analyze_multimodal_semantics(
            PNG,
            "image/png",
            task_context="转写手写公式",
            provider=provider,
        )


def test_remote_spec_is_bound_to_consent_region_and_retention_before_call(
    tmp_path,
) -> None:
    provider = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        result=_v2_result(),
        remote=True,
        region="eu_west",
        retention_days=7,
    )
    store = RemoteConsentStore(
        tmp_path / "consent-v2.json",
        signing_secret=b"m" * 32,
        clock=lambda: datetime(2026, 8, 12, tzinfo=timezone.utc),
    )
    wrong_region = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_visual_analysis",
        provider_id=provider.provider_id,
        processing_region="us_east",
        data_categories=["learner_image"],
        provider_retention_days=7,
    )
    with pytest.raises(VisualSemanticError, match="region"):
        analyze_multimodal_semantics(
            PNG,
            "image/png",
            task_context="描述几何图",
            provider=provider,
            subject_id="subject_abcdefghijklmnop",
            consent_id=wrong_region["consent_id"],
            consent_store=store,
        )
    assert provider.calls == 0

    wrong_retention = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_visual_analysis",
        provider_id=provider.provider_id,
        processing_region="eu_west",
        data_categories=["learner_image"],
        provider_retention_days=0,
    )
    with pytest.raises(VisualSemanticError, match="retention"):
        analyze_multimodal_semantics(
            PNG,
            "image/png",
            task_context="描述几何图",
            provider=provider,
            subject_id="subject_abcdefghijklmnop",
            consent_id=wrong_retention["consent_id"],
            consent_store=store,
        )
    assert provider.calls == 0

    receipt = store.grant(
        subject_id="subject_abcdefghijklmnop",
        purpose="remote_visual_analysis",
        provider_id=provider.provider_id,
        processing_region="eu_west",
        data_categories=["learner_image"],
        provider_retention_days=7,
    )
    evidence = analyze_multimodal_semantics(
        PNG,
        "image/png",
        task_context="描述几何图",
        provider=provider,
        subject_id="subject_abcdefghijklmnop",
        consent_id=receipt["consent_id"],
        consent_store=store,
    )
    assert provider.calls == 1
    assert evidence["privacy_receipt"]["processing_region"] == "eu_west"
    assert evidence["privacy_receipt"]["provider_retention_days"] == 7
    assert evidence["privacy_receipt"]["data_categories"] == ["learner_image"]


def test_timestamped_audio_is_local_only_and_unconfigured_provider_refuses() -> None:
    wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + b"local-audio"
    result = {
        "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
        "source_modality": "audio",
        "modality": "unknown",
        "transcription": {
            "status": "candidate",
            "text": "先定义状态 再写转移",
            "confidence": 0.94,
            "language": "zh-Hans",
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 850,
                    "text": "先定义状态",
                    "evidence_locator": "audio/0-850ms",
                    "confidence": 0.95,
                },
                {
                    "start_ms": 850,
                    "end_ms": 1800,
                    "text": "再写转移",
                    "evidence_locator": "audio/850-1800ms",
                    "confidence": 0.93,
                },
            ],
        },
        "semantic_analysis": {
            "status": "not_performed",
            "description": "仅执行本地时间戳转写。",
            "claims": [],
            "uncertainties": ["未核验转写内容的学科正确性"],
            "conflicts": [],
        },
        "decision": "abstain",
    }
    provider = _V2Provider(
        source_modalities=["audio"],
        mime_types=["audio/wav"],
        capabilities=["temporal_transcription", "transcription"],
        result=result,
    )
    with pytest.raises(VisualSemanticError, match="not configured"):
        analyze_temporal_media(
            wav,
            "audio/wav",
            task_context="转写",
            provider=None,
        )
    evidence = analyze_temporal_media(
        wav,
        "audio/wav",
        task_context="转写",
        provider=provider,
    )
    assert evidence["transcription"]["segments"][1]["start_ms"] == 850
    assert evidence["remote_media_sent"] is False
    assert evidence["decision"] == "abstain"

    remote = _V2Provider(
        source_modalities=["audio"],
        mime_types=["audio/wav"],
        capabilities=["temporal_transcription", "transcription"],
        result=result,
        remote=True,
        region="provider_managed",
        retention_days=0,
    )
    with pytest.raises(VisualSemanticError, match="must remain local"):
        analyze_temporal_media(
            wav,
            "audio/wav",
            task_context="转写",
            provider=remote,
        )
    assert remote.calls == 0


def test_scanned_pdf_layer_is_supported_locally_but_still_abstains() -> None:
    result = {
        "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
        "source_modality": "scanned_pdf_page",
        "modality": "handwriting",
        "transcription": {
            "status": "candidate",
            "text": "f(x)=x²",
            "confidence": 0.79,
            "language": "und",
            "segments": [],
        },
        "semantic_analysis": {
            "status": "abstained",
            "description": "扫描页含手写公式候选。",
            "claims": [],
            "uncertainties": ["上标边界不清晰"],
            "conflicts": [],
        },
        "decision": "abstain",
    }
    provider = _V2Provider(
        source_modalities=["scanned_pdf_page"],
        mime_types=["application/pdf"],
        capabilities=["semantic_analysis", "transcription"],
        result=result,
    )
    evidence = analyze_multimodal_semantics(
        b"%PDF-1.7\nscanned-page-fixture",
        "application/pdf",
        task_context="转写并标注扫描页的不确定部分",
        provider=provider,
    )
    assert evidence["source_modality"] == "scanned_pdf_page"
    assert evidence["modality"] == "handwriting"
    assert evidence["abstained"] is True
    assert evidence["grading_evidence_allowed"] is False


def test_v2_evidence_validates_against_the_public_stable_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    provider = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        result=_v2_result(),
    )
    evidence = analyze_multimodal_semantics(
        PNG,
        "image/png",
        task_context="描述几何图中的可见标记",
        provider=provider,
    )
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schema"
            / "teacher_agent_multimodal_evidence.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(evidence)
    tampered = json.loads(json.dumps(evidence))
    tampered["decision"] = "abstain"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(tampered)


def test_provider_spec_result_and_capability_surfaces_are_exact_and_fail_closed() -> None:
    provider = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        result=_v2_result(),
    )
    provider.provider_spec["undocumented_option"] = True
    with pytest.raises(VisualSemanticError, match="unsupported fields"):
        analyze_multimodal_semantics(
            PNG,
            "image/png",
            task_context="描述几何图",
            provider=provider,
        )
    assert provider.calls == 0

    provider = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        result={**_v2_result(), "raw_debug": "must not cross"},
    )
    with pytest.raises(VisualSemanticError, match="unsupported fields"):
        analyze_multimodal_semantics(
            PNG,
            "image/png",
            task_context="描述几何图",
            provider=provider,
        )

    provider = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        capabilities=["semantic_analysis"],
        result=_v2_result(),
    )
    with pytest.raises(VisualSemanticError, match="exceeds provider transcription"):
        analyze_multimodal_semantics(
            PNG,
            "image/png",
            task_context="描述几何图",
            provider=provider,
        )

    undeclared_retention = _V2Provider(
        source_modalities=["image"],
        mime_types=["image/png"],
        result=_v2_result(),
        remote=True,
        region="provider_managed",
        retention_days=None,
    )
    with pytest.raises(VisualSemanticError, match="retention policy"):
        analyze_multimodal_semantics(
            PNG,
            "image/png",
            task_context="描述几何图",
            provider=undeclared_retention,
        )
    assert undeclared_retention.calls == 0


def test_temporal_full_text_must_bind_exactly_to_ordered_segments() -> None:
    wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + b"local-audio"
    result = {
        "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
        "source_modality": "audio",
        "modality": "unknown",
        "transcription": {
            "status": "candidate",
            "text": "与时间段不一致",
            "confidence": 0.9,
            "language": "zh-Hans",
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 100,
                    "text": "真实片段",
                    "evidence_locator": "audio/0-100ms",
                    "confidence": 0.9,
                }
            ],
        },
        "semantic_analysis": {
            "status": "not_performed",
            "description": "仅转写。",
            "claims": [],
            "uncertainties": ["语义未核验"],
            "conflicts": [],
        },
        "decision": "abstain",
    }
    provider = _V2Provider(
        source_modalities=["audio"],
        mime_types=["audio/wav"],
        capabilities=["temporal_transcription", "transcription"],
        result=result,
    )
    with pytest.raises(VisualSemanticError, match="diverges"):
        analyze_temporal_media(
            wav,
            "audio/wav",
            task_context="转写",
            provider=provider,
        )


def test_rendered_slide_analysis_preserves_slide_notes_conflict_as_unresolved() -> None:
    conflict = {
        "kind": "slide_notes_disagreement",
        "description": "正文写阈值 8，备注写阈值 6。",
        "evidence_locators": ["slide/body", "slide/notes"],
    }
    provider = _V2Provider(
        source_modalities=["slide"],
        mime_types=["image/png"],
        result=_v2_result(
            source="slide",
            modality="mixed",
            decision="requires_confirmation",
            conflicts=[conflict],
        ),
    )
    evidence = analyze_multimodal_semantics(
        PNG,
        "image/png",
        task_context="分别核对幻灯片正文和讲者备注",
        provider=provider,
        source_modality="slide",
    )
    assert evidence["source_modality"] == "slide"
    assert evidence["decision"] == "requires_confirmation"
    assert evidence["conflicts"][0]["kind"] == "slide_notes_disagreement"
    assert evidence["grading_evidence_allowed"] is False
