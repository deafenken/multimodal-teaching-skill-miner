from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import http.client
import io
import json
from pathlib import Path
import threading
import wave
import zipfile
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest

from teaching_skill_miner.cli import main
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    _record_store_value,
    _refresh_integrity,
    _teaching_resource_metadata,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)
from teaching_skill_miner.teacher_agent_multimodal import (
    MULTIMODAL_PROVIDER_RESULT_SCHEMA,
    MULTIMODAL_PROVIDER_SPEC_SCHEMA,
)
from teaching_skill_miner.teacher_agent_resources import extract_teaching_resource


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "data" / "teacher_agent_skill_library_v2.json"
DEMO_INPUT = ROOT / "data" / "teacher_agent_demo_input.json"
CASES = ROOT / "data" / "teacher_agent_evaluation_cases.json"


def _wav_fixture() -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\x00\x00" * 8_000)
    return target.getvalue()


def _conflicting_pptx_fixture() -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>阈值为 8</a:t></p:sld>',
        )
        archive.writestr(
            "ppt/slides/_rels/slide1.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="r1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" '
            'Target="../notesSlides/notesSlide1.xml" />'
            "</Relationships>",
        )
        archive.writestr(
            "ppt/notesSlides/notesSlide1.xml",
            '<p:notes xmlns:p="urn:p" xmlns:a="urn:a">'
            "<a:t>最终阈值应为 6</a:t></p:notes>",
        )
    return target.getvalue()


class _LocalTemporalProvider:
    provider_id = "local-dashboard-transcriber-v1"
    processing_region = "on_device"
    sends_raw_media_remotely = False

    def __init__(self) -> None:
        self.calls = 0
        self.private_token = "never-publish-provider-secret"
        self.provider_spec = {
            "schema": MULTIMODAL_PROVIDER_SPEC_SCHEMA,
            "provider_id": self.provider_id,
            "adapter_version": "1.0.0",
            "execution_scope": "local",
            "processing_region": self.processing_region,
            "raw_media_transport": "none",
            "provider_retention_days": 0,
            "supported_source_modalities": ["audio"],
            "supported_mime_types": ["audio/wav"],
            "capabilities": ["temporal_transcription", "transcription"],
            "deterministic": True,
        }

    def analyze(self, media_bytes, mime_type, *, task_context):
        self.calls += 1
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
                "description": "只执行本地时间戳转写。",
                "claims": [],
                "uncertainties": ["学科语义未核验"],
                "conflicts": [],
            },
            "decision": "abstain",
        }


class _RemoteTemporalProvider(_LocalTemporalProvider):
    processing_region = "provider_managed"
    sends_raw_media_remotely = True

    def __init__(self) -> None:
        super().__init__()
        self.provider_spec.update(
            {
                "execution_scope": "remote",
                "processing_region": self.processing_region,
                "raw_media_transport": "remote",
                "provider_retention_days": 0,
            }
        )


class _DashboardHTTP:
    def __init__(self, snapshot) -> None:
        try:
            self.server, self.url = create_teacher_agent_dashboard_server(
                snapshot, capability_token="m" * 24
            )
        except PermissionError:
            pytest.skip("sandbox does not permit loopback socket binding")
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.parsed = urlsplit(self.url)

    def request(self, method: str, route: str, body=None) -> tuple[int, dict]:
        connection = http.client.HTTPConnection(
            self.parsed.hostname, self.parsed.port, timeout=5
        )
        headers: dict[str, str] = {}
        encoded = None
        if body is not None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(
            method,
            f"{self.parsed.path}{route}",
            body=encoded,
            headers=headers,
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        status = response.status
        connection.close()
        return status, payload

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)


def _snapshot(**kwargs):
    return build_teacher_agent_dashboard_snapshot(
        LIBRARY,
        DEMO_INPUT,
        CASES,
        **kwargs,
    )


def _resource_body(data: bytes, *, name: str, mime: str, key: str) -> dict:
    return {
        "resource_idempotency_key": key,
        "mime_type": mime,
        "display_name": name,
        "data_base64": base64.b64encode(data).decode("ascii"),
    }


def _canonical_sha256(value: dict) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_bootstrap_publishes_stable_redacted_temporal_provider_spec_over_http() -> None:
    provider = _LocalTemporalProvider()
    snapshot = _snapshot(temporal_transcription_provider=provider)
    client = _DashboardHTTP(snapshot)
    try:
        status, payload = client.request("GET", "api/bootstrap")
    finally:
        client.close()
    assert status == 200
    public = payload["temporal_transcription"]
    assert public["available"] is True
    assert public["provider_spec"] == provider.provider_spec | {
        "compatibility_adapter": False
    }
    assert public["provider_spec_sha256"] == _canonical_sha256(
        public["provider_spec"]
    )
    assert public["local_only_under_current_consent_policy"] is True
    assert public["remote_audio_video_authorized"] is False
    assert public["credentials_exposed"] is False
    assert provider.private_token not in json.dumps(payload, ensure_ascii=False)


def test_resource_http_routes_temporal_provider_but_abstention_never_enters_session(
    tmp_path,
) -> None:
    provider = _LocalTemporalProvider()
    snapshot = _snapshot(
        temporal_transcription_provider=provider,
        store_path=tmp_path / "sessions.jsonl",
        resource_index_store_path=tmp_path / "resource-index",
    )
    client = _DashboardHTTP(snapshot)
    try:
        status, staged = client.request(
            "POST",
            "api/resource",
            _resource_body(
                _wav_fixture(),
                name="讲解.wav",
                mime="audio/wav",
                key="temporal-http-stage-001",
            ),
        )
        assert status == 200
        assert provider.calls == 1
        assert staged["session_use"]["status"] == "blocked_abstained"
        assert staged["session_use"]["grading_evidence_allowed"] is False
        assert staged["resource"]["temporal_transcription_receipt"][
            "remote_media_sent"
        ] is False
        assert staged["resource"]["evidence_contract"]["temporal_provenance"][0][
            "evidence_locator"
        ] == "audio/0-900ms"
        assert "extracted_text" not in staged["resource"]

        status, rejected = client.request(
            "POST",
            "api/start",
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "temporal-http-start-001",
                "staged_resource_ids": [
                    staged["resource"]["staged_resource_id"]
                ],
            },
        )
        assert status == 400
        assert "abstains from unverified semantics" in rejected["error"]
        assert snapshot.sessions == {}

        status, started = client.request(
            "POST",
            "api/start",
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "profile_revision": "temporal-http-profile-v1",
                "start_idempotency_key": "temporal-http-clean-start-001",
            },
        )
        assert status == 200
        context_version = started["context_version"]
        status, active_rejected = client.request(
            "POST",
            "api/resource",
            {
                **_resource_body(
                    _wav_fixture(),
                    name="补充讲解.wav",
                    mime="audio/wav",
                    key="temporal-http-active-001",
                ),
                "session_id": started["session_id"],
                "expected_round": started["rounds_completed"],
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": context_version,
                "profile_revision": "temporal-http-profile-v1",
            },
        )
        assert status == 400
        assert "abstains from unverified semantics" in active_rejected["error"]
        record = snapshot.sessions[started["session_id"]]
        assert record.context_version == context_version
        assert record.teaching_resources == {}
        assert record.session.get("teaching_resources", []) == []
    finally:
        client.close()


def test_resource_http_stages_conflict_but_start_requires_confirmation() -> None:
    snapshot = _snapshot()
    client = _DashboardHTTP(snapshot)
    try:
        status, staged = client.request(
            "POST",
            "api/resource",
            _resource_body(
                _conflicting_pptx_fixture(),
                name="冲突课件.pptx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.presentation"
                ),
                key="conflict-http-stage-001",
            ),
        )
        assert status == 200
        assert staged["session_use"]["status"] == "blocked_pending_confirmation"
        assert staged["resource"]["requires_confirmation"] is True
        status, rejected = client.request(
            "POST",
            "api/start",
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "conflict-http-start-001",
                "staged_resource_ids": [
                    staged["resource"]["staged_resource_id"]
                ],
            },
        )
        assert status == 400
        assert "teacher confirmation is required" in rejected["error"]
        assert snapshot.sessions == {}
    finally:
        client.close()


def test_unconfigured_temporal_upload_is_honestly_rejected_over_http() -> None:
    snapshot = _snapshot()
    client = _DashboardHTTP(snapshot)
    try:
        status, rejected = client.request(
            "POST",
            "api/resource",
            _resource_body(
                _wav_fixture(),
                name="未配置.wav",
                mime="audio/wav",
                key="unconfigured-temporal-http-001",
            ),
        )
    finally:
        client.close()
    assert status == 400
    assert "转写适配器未配置" in rejected["error"]


def test_factory_rejects_remote_temporal_provider_before_any_media_dispatch() -> None:
    provider = _RemoteTemporalProvider()
    with pytest.raises(TeacherAgentDashboardError, match="must remain local"):
        _snapshot(temporal_transcription_provider=provider)
    assert provider.calls == 0


def test_cli_passes_explicit_temporal_provider_factory_to_dashboard() -> None:
    provider = _LocalTemporalProvider()
    captured: dict = {}

    def fake_serve(*args, **kwargs):
        captured.update(kwargs)
        return 0

    with (
        patch(
            "teaching_skill_miner.cli._load_temporal_transcription_provider",
            return_value=provider,
        ) as loader,
        patch("teaching_skill_miner.cli.serve_teacher_agent_dashboard", fake_serve),
    ):
        result = main(
            [
                "teacher-agent-dashboard",
                "--agent-backend",
                "deterministic",
                "--no-browser",
                "--temporal-transcription-provider-factory",
                "deployment.providers:build_local_asr",
            ]
        )
    assert result == 0
    loader.assert_called_once_with("deployment.providers:build_local_asr")
    assert captured["temporal_transcription_provider"] is provider


def test_provider_spec_mutation_after_startup_fails_closed_before_transcription() -> None:
    provider = _LocalTemporalProvider()
    snapshot = _snapshot(temporal_transcription_provider=provider)
    frozen = deepcopy(snapshot.temporal_transcription_provider_spec)
    provider.provider_spec["adapter_version"] = "2.0.0"
    with pytest.raises(
        TeacherAgentDashboardError, match="spec changed after startup"
    ):
        snapshot.upload_resource(
            _resource_body(
                _wav_fixture(),
                name="变更适配器.wav",
                mime="audio/wav",
                key="mutated-temporal-provider-001",
            )
        )
    assert snapshot.temporal_transcription_provider_spec == frozen
    assert provider.calls == 0


def test_durable_recovery_refuses_preexisting_abstained_resource_context() -> None:
    provider = _LocalTemporalProvider()
    snapshot = _snapshot(temporal_transcription_provider=provider)
    started = snapshot.start(
        {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "recovery-abstention-start-001",
        }
    )
    record = snapshot.sessions[started["session_id"]]
    resource = extract_teaching_resource(
        _wav_fixture(),
        "audio/wav",
        display_name="旧会话讲解.wav",
        temporal_transcription_provider=provider,
    )
    record.session["teaching_resources"] = [resource]
    _refresh_integrity(record.session)
    record.teaching_resources[resource["resource_id"]] = (
        _teaching_resource_metadata(resource)
    )
    stored = _record_store_value(record)
    with pytest.raises(
        TeacherAgentDashboardError,
        match="now requires confirmation or abstention review",
    ):
        snapshot._record_from_recovery_value(stored)
