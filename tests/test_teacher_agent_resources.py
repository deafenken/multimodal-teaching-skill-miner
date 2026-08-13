from __future__ import annotations

import io
from copy import deepcopy
from types import SimpleNamespace
import wave
import zipfile

import pytest

from teaching_skill_miner.teacher_agent_resources import (
    MAX_RESOURCE_TEXT_CHARS,
    TeachingResourceError,
    extract_teaching_resource,
    inspect_temporal_media_metadata,
    runtime_supported_resource_extensions,
    teaching_resource_for_session,
    validate_teaching_resources,
)
from teaching_skill_miner.teacher_agent_multimodal import (
    MULTIMODAL_PROVIDER_RESULT_SCHEMA,
    MULTIMODAL_PROVIDER_SPEC_SCHEMA,
)
from teaching_skill_miner.teacher_agent_resource_retrieval import (
    TeachingResourceIndexStore,
)


def _office_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return buffer.getvalue()


def test_runtime_resource_capabilities_only_advertise_available_extractors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "teaching_skill_miner.teacher_agent_resources.shutil.which",
        lambda command: "/usr/bin/pdftotext" if command == "pdftotext" else None,
    )
    monkeypatch.setattr(
        "teaching_skill_miner.teacher_agent_resources.local_visual_extractor_available",
        lambda: False,
    )

    formats = runtime_supported_resource_extensions()

    assert {
        "txt",
        "md",
        "markdown",
        "csv",
        "tsv",
        "xlsx",
        "docx",
        "pptx",
        "pdf",
    } <= set(formats)
    assert {
        "doc",
        "rtf",
        "ppt",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "wav",
        "mp3",
        "mp4",
    }.isdisjoint(formats)

    with_temporal = runtime_supported_resource_extensions(
        temporal_transcription_provider=_LocalTemporalProvider(),
    )
    assert {"wav", "mp3", "m4a", "webm", "mp4", "mov"} <= set(with_temporal)

    with_invalid_temporal = runtime_supported_resource_extensions(
        temporal_transcription_provider=object(),  # type: ignore[arg-type]
    )
    assert {"wav", "mp3", "m4a", "webm", "mp4", "mov"}.isdisjoint(with_invalid_temporal)


def test_extract_text_resource_is_bounded_and_never_retains_raw_media() -> None:
    payload = ("教学目标：理解状态转移。\n" + "练习" * 8000).encode("utf-8")

    resource = extract_teaching_resource(
        payload,
        "text/plain",
        display_name="lesson.md",
    )

    assert resource["resource_type"] == "text"
    assert resource["extracted_text"].startswith("教学目标")
    assert resource["extracted_char_count"] == MAX_RESOURCE_TEXT_CHARS
    assert resource["truncated"] is True
    assert resource["raw_media_retained"] is False
    assert resource["remote_media_sent"] is False


def test_extract_docx_uses_ordered_document_xml_text() -> None:
    payload = _office_zip(
        {
            "word/document.xml": (
                '<w:document xmlns:w="urn:w"><w:body>'
                "<w:p><w:r><w:t>第一段</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>第二段</w:t></w:r></w:p>"
                "</w:body></w:document>"
            )
        }
    )

    resource = extract_teaching_resource(
        payload,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        display_name="教学文稿.docx",
    )

    assert resource["resource_type"] == "document"
    assert "第一段" in resource["extracted_text"]
    assert "第二段" in resource["extracted_text"]
    assert resource["extraction_engine"] == "stdlib_docx_xml"


def test_extract_pptx_preserves_slide_order_and_count() -> None:
    payload = _office_zip(
        {
            "ppt/slides/slide2.xml": '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>第二页</a:t></p:sld>',
            "ppt/slides/slide1.xml": '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>第一页</a:t></p:sld>',
        }
    )

    resource = extract_teaching_resource(
        payload,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        display_name="课程.pptx",
    )

    assert resource["resource_type"] == "presentation"
    assert resource["page_count"] == 2
    assert resource["extracted_text"].index("第一页") < resource[
        "extracted_text"
    ].index("第二页")
    assert resource["extraction_engine"] == "stdlib_pptx_xml_with_notes"


def test_extract_pptx_includes_speaker_notes_and_flags_visual_slide(tmp_path) -> None:
    payload = _office_zip(
        {
            "ppt/slides/slide1.xml": (
                '<p:sld xmlns:p="urn:p" xmlns:a="urn:a">'
                "<a:t>光合作用概览</a:t><p:pic /></p:sld>"
            ),
            "ppt/slides/_rels/slide1.xml.rels": (
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="r1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" '
                'Target="../notesSlides/notesSlide1.xml" />'
                '<Relationship Id="r2" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                'Target="../media/image1.png" />'
                "</Relationships>"
            ),
            "ppt/notesSlides/notesSlide1.xml": (
                '<p:notes xmlns:p="urn:p" xmlns:a="urn:a">'
                "<a:t>提醒学生区分光反应与暗反应。</a:t></p:notes>"
            ),
        }
    )

    store = TeachingResourceIndexStore(tmp_path / "resources")
    resource = extract_teaching_resource(
        payload,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        display_name="生物课.pptx",
        index_store=store,
    )

    assert "[讲者备注]" in resource["extracted_text"]
    assert "提醒学生" in resource["extracted_text"]
    assert "视觉复核" in resource["extracted_text"]
    document = store.get_index_document(resource["content_sha256"])
    assert document is not None
    chunks = document["retrieval_index"]["chunks"]
    assert any(chunk["content_kind"] == "speaker_notes" for chunk in chunks)
    assert any(chunk["needs_visual_review"] is True for chunk in chunks)


def test_extract_pptx_recovers_structured_table_chart_and_formula_without_visual_claims(
    tmp_path,
) -> None:
    payload = _office_zip(
        {
            "ppt/slides/slide1.xml": (
                '<p:sld xmlns:p="urn:p" xmlns:a="urn:a" xmlns:m="urn:m">'
                "<a:t>实验记录</a:t>"
                "<a:tbl><a:tr><a:tc><a:t>条件</a:t></a:tc>"
                "<a:tc><a:t>结果</a:t></a:tc></a:tr>"
                "<a:tr><a:tc><a:t>有光</a:t></a:tc>"
                "<a:tc><a:t>8</a:t></a:tc></a:tr></a:tbl>"
                "<m:oMath><m:r><m:t>E = mc²</m:t></m:r></m:oMath>"
                "</p:sld>"
            ),
            "ppt/slides/_rels/slide1.xml.rels": (
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="r1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
                'Target="../charts/chart1.xml" />'
                "</Relationships>"
            ),
            "ppt/charts/chart1.xml": (
                '<c:chartSpace xmlns:c="urn:c"><c:barChart><c:ser>'
                "<c:tx><c:strRef><c:strCache><c:pt><c:v>产氧量</c:v></c:pt>"
                "</c:strCache></c:strRef></c:tx>"
                "<c:cat><c:strRef><c:strCache>"
                "<c:pt><c:v>弱光</c:v></c:pt><c:pt><c:v>强光</c:v></c:pt>"
                "</c:strCache></c:strRef></c:cat>"
                "<c:val><c:numRef><c:numCache>"
                "<c:pt><c:v>2</c:v></c:pt><c:pt><c:v>8</c:v></c:pt>"
                "</c:numCache></c:numRef></c:val>"
                "</c:ser></c:barChart></c:chartSpace>"
            ),
        }
    )

    store = TeachingResourceIndexStore(tmp_path / "resources")
    resource = extract_teaching_resource(
        payload,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        display_name="实验图表.pptx",
        index_store=store,
    )

    text = resource["extracted_text"]
    assert "结构化表格" in text
    assert "行 2: 有光 | 8" in text
    assert "公式 OOXML 文字转写候选" in text
    assert "E = mc²" in text
    assert "结构化图表数据" in text
    assert "类别=弱光 | 强光" in text
    assert "数值=2 | 8" in text
    assert "不代表已理解坐标轴" in text
    assert resource["needs_review"] is True
    document = store.get_index_document(resource["content_sha256"])
    assert document is not None
    assert all(
        chunk["needs_visual_review"] is True
        for chunk in document["retrieval_index"]["chunks"]
    )


def test_rejects_unsupported_or_disguised_resource() -> None:
    with pytest.raises(TeachingResourceError, match="暂不支持"):
        extract_teaching_resource(
            b"hello", "application/octet-stream", display_name="lesson.zip"
        )
    with pytest.raises(TeachingResourceError, match="有效 PDF"):
        extract_teaching_resource(
            b"not a pdf", "application/pdf", display_name="lesson.pdf"
        )


def test_csv_and_xlsx_preserve_cells_but_never_promote_formulas_to_truth() -> None:
    csv_resource = extract_teaching_resource(
        "项目,数值\n甲,=1+1\n".encode(),
        "text/csv",
        display_name="实验.csv",
    )
    assert csv_resource["resource_type"] == "spreadsheet"
    assert "行 2: 甲 | =1+1" in csv_resource["extracted_text"]
    assert "公式计算结果与引用关系未核验" in csv_resource["extracted_text"]
    assert csv_resource["grading_evidence_allowed"] is False
    assert csv_resource["mastery_evidence_allowed"] is False

    xlsx = _office_zip(
        {
            "xl/sharedStrings.xml": (
                '<sst xmlns="urn:x"><si><t>项目</t></si><si><t>甲</t></si></sst>'
            ),
            "xl/worksheets/sheet1.xml": (
                '<worksheet xmlns="urn:x"><sheetData>'
                '<row r="1"><c r="A1" t="s"><v>0</v></c>'
                '<c r="B1"><v>数值</v></c></row>'
                '<row r="2"><c r="A2" t="s"><v>1</v></c>'
                '<c r="B2"><f>1+1</f><v>2</v></c></row>'
                "</sheetData></worksheet>"
            ),
        }
    )
    xlsx_resource = extract_teaching_resource(
        xlsx,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        display_name="实验.xlsx",
    )
    assert "B2=公式=1+1；缓存值=2" in xlsx_resource["extracted_text"]
    assert "缓存值不是重新计算结果" in xlsx_resource["extracted_text"]
    kinds = {layer["kind"] for layer in xlsx_resource["evidence_contract"]["layers"]}
    assert {"spreadsheet_cells", "formula_transcription"} <= kinds
    assert xlsx_resource["evidence_contract"]["decision"] == (
        "abstain_from_unverified_semantics"
    )


def test_slide_notes_numeric_conflict_requires_confirmation(tmp_path) -> None:
    payload = _office_zip(
        {
            "ppt/slides/slide1.xml": (
                '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>阈值为 8</a:t></p:sld>'
            ),
            "ppt/slides/_rels/slide1.xml.rels": (
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="r1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" '
                'Target="../notesSlides/notesSlide1.xml" />'
                "</Relationships>"
            ),
            "ppt/notesSlides/notesSlide1.xml": (
                '<p:notes xmlns:p="urn:p" xmlns:a="urn:a">'
                "<a:t>最终阈值应为 6</a:t></p:notes>"
            ),
        }
    )
    store = TeachingResourceIndexStore(tmp_path / "conflicting-slide-index")
    resource = extract_teaching_resource(
        payload,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        display_name="冲突课件.pptx",
        index_store=store,
    )
    assert "内容冲突待确认" in resource["extracted_text"]
    assert resource["requires_confirmation"] is True
    contract = resource["evidence_contract"]
    assert contract["decision"] == "requires_confirmation"
    assert contract["conflicts"][0]["kind"] == "slide_notes_disagreement"
    assert contract["mastery_evidence_allowed"] is False
    document = store.get_index_document(resource["content_sha256"])
    assert document is not None
    assert all(
        chunk["needs_visual_review"] is True
        for chunk in document["retrieval_index"]["chunks"]
    )
    with pytest.raises(TeachingResourceError, match="teacher confirmation"):
        teaching_resource_for_session(resource)


def test_scanned_pdf_without_embedded_text_is_a_visible_abstention(
    monkeypatch,
) -> None:
    from teaching_skill_miner import teacher_agent_resources as resources_module

    monkeypatch.setattr(
        resources_module.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(
        resources_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"\f", stderr=b""),
    )
    resource = extract_teaching_resource(
        b"%PDF-1.7\nfixture",
        "application/pdf",
        display_name="扫描讲义.pdf",
    )
    assert resource["page_count"] == 1
    assert "扫描页检测" in resource["extracted_text"]
    assert "当前必须弃权" in resource["extracted_text"]
    assert resource["evidence_contract"]["decision"] == (
        "abstain_from_unverified_semantics"
    )
    assert resource["grading_evidence_allowed"] is False


def _wav_fixture() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(8_000)
        target.writeframes(b"\x00\x00" * 8_000)
    return buffer.getvalue()


class _LocalTemporalProvider:
    provider_id = "local-test-transcriber-v1"
    processing_region = "on_device"
    sends_raw_media_remotely = False
    provider_spec = {
        "schema": MULTIMODAL_PROVIDER_SPEC_SCHEMA,
        "provider_id": provider_id,
        "adapter_version": "1.0.0",
        "execution_scope": "local",
        "processing_region": processing_region,
        "raw_media_transport": "none",
        "provider_retention_days": 0,
        "supported_source_modalities": ["audio"],
        "supported_mime_types": ["audio/wav"],
        "capabilities": ["temporal_transcription", "transcription"],
        "deterministic": True,
    }

    def analyze(self, media_bytes, mime_type, *, task_context):
        assert media_bytes.startswith(b"RIFF")
        assert mime_type == "audio/wav"
        assert "时间" in task_context
        return {
            "schema": MULTIMODAL_PROVIDER_RESULT_SCHEMA,
            "source_modality": "audio",
            "modality": "unknown",
            "transcription": {
                "status": "candidate",
                "text": "先定义状态",
                "confidence": 0.96,
                "language": "zh-Hans",
                "segments": [
                    {
                        "start_ms": 0,
                        "end_ms": 900,
                        "text": "先定义状态",
                        "evidence_locator": "audio/0-900ms",
                        "confidence": 0.96,
                    }
                ],
            },
            "semantic_analysis": {
                "status": "not_performed",
                "description": "只执行转写。",
                "claims": [],
                "uncertainties": ["学科语义未核验"],
                "conflicts": [],
            },
            "decision": "abstain",
        }


def test_audio_metadata_and_timestamped_local_transcription_are_provenanced() -> None:
    wav = _wav_fixture()
    metadata = inspect_temporal_media_metadata(
        wav, "audio/wav", display_name="讲解.wav"
    )
    assert metadata["metadata_status"] == "parsed"
    assert metadata["duration_ms"] == 1_000
    assert metadata["transcription_performed"] is False
    with pytest.raises(TeachingResourceError, match="转写适配器未配置"):
        extract_teaching_resource(
            wav,
            "audio/wav",
            display_name="讲解.wav",
        )

    resource = extract_teaching_resource(
        wav,
        "audio/wav",
        display_name="讲解.wav",
        temporal_transcription_provider=_LocalTemporalProvider(),
    )
    assert resource["resource_type"] == "audio"
    assert "00:00:00.000-00:00:00.900" in resource["extracted_text"]
    assert resource["temporal_metadata"]["transcription_performed"] is True
    assert resource["evidence_contract"]["temporal_provenance"] == [
        {
            "segment_id": "segment_0001",
            "start_ms": 0,
            "end_ms": 900,
            "evidence_locator": "audio/0-900ms",
            "confidence": 0.96,
        }
    ]
    assert resource["temporal_transcription_receipt"]["remote_media_sent"] is False
    assert resource["mastery_evidence_allowed"] is False
    validate_teaching_resources([resource])

    tampered = deepcopy(resource)
    tampered["evidence_contract"]["mastery_evidence_allowed"] = True
    with pytest.raises(TeachingResourceError, match="assessment boundary"):
        validate_teaching_resources([tampered])

    beyond_duration = deepcopy(resource)
    beyond_duration["evidence_contract"]["temporal_provenance"][0]["end_ms"] = 3_000
    with pytest.raises(TeachingResourceError, match="exceeds media duration"):
        validate_teaching_resources([beyond_duration])


def test_video_container_metadata_is_local_but_does_not_fake_a_transcript(
    monkeypatch,
) -> None:
    from teaching_skill_miner import teacher_agent_resources as resources_module

    monkeypatch.setattr(resources_module.shutil, "which", lambda name: None)
    video = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
    metadata = inspect_temporal_media_metadata(
        video, "video/mp4", display_name="讲解.mp4"
    )
    assert metadata["source_modality"] == "video"
    assert metadata["metadata_status"] == "container_signature_only"
    assert metadata["duration_ms"] is None
    assert metadata["transcription_performed"] is False
    with pytest.raises(TeachingResourceError, match="元数据不能代替转写"):
        extract_teaching_resource(
            video,
            "video/mp4",
            display_name="讲解.mp4",
        )


def test_ambiguous_webm_and_ooxml_traversal_members_fail_closed() -> None:
    with pytest.raises(TeachingResourceError, match="不能从扩展名猜测"):
        extract_teaching_resource(
            b"\x1aE\xdf\xa3fixture",
            "application/octet-stream",
            display_name="讲解.webm",
        )

    malicious = _office_zip(
        {
            "word/document.xml": (
                '<w:document xmlns:w="urn:w"><w:body><w:p>'
                "<w:r><w:t>正文</w:t></w:r></w:p></w:body></w:document>"
            ),
            "../outside.txt": "must never be trusted",
        }
    )
    with pytest.raises(TeachingResourceError, match="压缩部件不安全"):
        extract_teaching_resource(
            malicious,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            display_name="恶意.docx",
        )
