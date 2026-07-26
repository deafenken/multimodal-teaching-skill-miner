"""Environment diagnostics for reproducible local and packaged execution."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

from .io_utils import resource_root


def _package_status(distribution: str, module: str) -> dict[str, Any]:
    available = importlib.util.find_spec(module) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": available, "version": version}


def _command_status(
    env_name: str,
    default: str,
    *,
    version_args: tuple[str, ...] = ("--version",),
) -> dict[str, Any]:
    configured = os.getenv(env_name, default).strip() or default
    executable = shutil.which(configured)
    version = None
    error = None
    if executable:
        try:
            result = subprocess.run(
                [executable, *version_args],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            lines = (result.stdout or result.stderr).splitlines()
            version = lines[0].strip() if lines else None
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = type(exc).__name__
    return {
        "configured_command": configured,
        "available": executable is not None,
        "executable": executable,
        "version": version,
        "probe_error": error,
    }


def _whisper_status() -> dict[str, Any]:
    configured = os.getenv("TSM_WHISPER", "whisper").strip() or "whisper"
    candidates = list(dict.fromkeys([configured, "whisper", "whisper-cli", "main"]))
    configured_executable = shutil.which(configured)
    discovered = next((shutil.which(value) for value in candidates if shutil.which(value)), None)
    compatible = bool(
        configured_executable
        and Path(configured_executable).name.lower() in {"whisper", "whisper.exe"}
    )
    return {
        "configured_command": configured,
        "available": configured_executable is not None,
        "executable": configured_executable,
        "openai_whisper_cli_compatible": compatible,
        "different_whisper_cli_discovered": (
            discovered if discovered and discovered != configured_executable else None
        ),
        "searched_commands": candidates,
        "note": "The current preprocessor supports the OpenAI Whisper CLI interface. whisper.cpp's whisper-cli/main is detected but not reported ready because its arguments and model files differ.",
    }


def _resource_status() -> dict[str, Any]:
    try:
        root = resource_root()
    except FileNotFoundError as exc:
        return {
            "available": False,
            "root": None,
            "missing": ["bundled resource root"],
            "error": str(exc),
        }
    manifest = root / "data" / "dataset_manifest.json"
    transcripts = sorted((root / "data" / "transcripts").glob("*.json"))
    governance_root = root / "governance" if (root / "governance").is_dir() else root
    required = [
        manifest,
        root / "data" / "evaluation_cases.json",
        root / "data" / "formal_caption_sources.json",
        root / "schema" / "teaching_skill.schema.json",
        root / "schema" / "formal_caption_sources.schema.json",
        root / "schema" / "multimodal_event.schema.json",
        root / "schema" / "strict_recognition_manifest.schema.json",
        root / "schema" / "strict_feature_bundle.schema.json",
        root / "schema" / "raw_feature_extractor_config.schema.json",
        root / "schema" / "external_claim_contract.schema.json",
        root / "schema" / "frozen_recognition_model.schema.json",
        root / "schema" / "freeze_registration_request.schema.json",
        root / "schema" / "freeze_registration_attestation.schema.json",
        root / "schema" / "external_evaluation_receipt.schema.json",
        root / "schema" / "external_research_evidence.schema.json",
        root / "schema" / "external_research_evidence_attestation.schema.json",
        root / "configs" / "external_claim_contract.example.json",
        root / "configs" / "raw_feature_extractor.example.json",
        governance_root / "CHANGELOG.md",
        governance_root / "PRIVACY.md",
        governance_root / "SECURITY.md",
        governance_root / "THIRD_PARTY_DATA.md",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if len(transcripts) < 10:
        missing.append(f"data/transcripts/*.json (found {len(transcripts)}, need at least 10)")
    return {
        "available": not missing,
        "root": str(root),
        "transcript_count": len(transcripts),
        "missing": missing,
        "error": None,
    }


def _api_status() -> dict[str, Any]:
    base = os.getenv("TSM_API_BASE", "https://api.openai.com/v1").strip()
    parsed = urlparse(base)
    localhost = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    transport_safe = parsed.scheme == "https" or (parsed.scheme == "http" and localhost)
    return {
        "configured": bool(os.getenv("TSM_API_KEY", "").strip()),
        "base_url": base,
        "model": os.getenv("TSM_MODEL", "gpt-4.1-mini"),
        "transport_safe": transport_safe,
        "key_value_exposed": False,
    }


def doctor_report() -> dict[str, Any]:
    """Return a secret-safe capability report without modifying the environment."""

    python_ready = sys.version_info >= (3, 10)
    resources = _resource_status()
    packages = {
        "numpy": _package_status("numpy", "numpy"),
        "scikit_learn": _package_status("scikit-learn", "sklearn"),
        "cryptography": _package_status("cryptography", "cryptography"),
    }
    tools = {
        "ffmpeg": _command_status("TSM_FFMPEG", "ffmpeg", version_args=("-version",)),
        "ffprobe": _command_status("TSM_FFPROBE", "ffprobe", version_args=("-version",)),
        "tesseract": _command_status("TSM_TESSERACT", "tesseract"),
        "whisper": _whisper_status(),
    }
    api = _api_status()
    target = Path.cwd()
    try:
        free_bytes = shutil.disk_usage(target).free
    except OSError:
        free_bytes = None
    writable = os.access(target, os.W_OK)
    core_ready = python_ready and bool(resources["available"])
    recognition_ready = packages["numpy"]["available"] and packages["scikit_learn"]["available"]
    attestation_ready = packages["cryptography"]["available"]
    multimodal_ready = tools["ffmpeg"]["available"] and tools["ffprobe"]["available"]
    asr_ready = (
        multimodal_ready
        and tools["whisper"]["available"]
        and tools["whisper"]["openai_whisper_cli_compatible"]
    )
    report = {
        "schema_version": "1.0",
        "status": "ready" if core_ready else "blocked",
        "status_scope": (
            "offline_core_and_bundled_resources_only; inspect every capability flag "
            "before claiming media ASR, recognition, API, or deployment readiness"
        ),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "supported": python_ready,
            "minimum": "3.10",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "resources": resources,
        "packages": packages,
        "external_tools": tools,
        "api_backend": api,
        "workspace": {
            "path": str(target.resolve()),
            "writable": writable,
            "free_bytes": free_bytes,
        },
        "capabilities": {
            "offline_demo_ready": core_ready and writable,
            "transcript_only_pipeline_ready": core_ready and writable,
            "multimodal_analysis_ready": core_ready and multimodal_ready and writable,
            "ocr_ready": core_ready and multimodal_ready and tools["tesseract"]["available"],
            "media_transcription_ready": core_ready and asr_ready and writable,
            "formal_caption_retrieval_prerequisites_ready": (
                core_ready and tools["ffprobe"]["available"] and writable
            ),
            "recognition_experiments_ready": core_ready and recognition_ready,
            "raw_sensor_feature_bridge_ready": (
                core_ready and packages["numpy"]["available"] and writable
            ),
            "raw_media_feature_bridge_ready": (
                core_ready
                and packages["numpy"]["available"]
                and multimodal_ready
                and writable
            ),
            "signed_external_evaluation_ready": (
                core_ready and recognition_ready and attestation_ready and writable
            ),
            "external_research_evidence_attestation_ready": (
                core_ready and attestation_ready and writable
            ),
            "api_refinement_ready": core_ready and api["configured"] and api["transport_safe"],
        },
        "limitations": [
            "Top-level status=ready means only that the offline core and bundled resources are available; it is not full-pipeline, ASR, deployment, privacy, CI, or release readiness.",
            "Tool presence does not prove model files, language packs, codecs, or GPU capacity; run the relevant smoke test.",
            "API readiness never prints the configured secret and does not authorize transcript upload.",
            "Formal-caption retrieval prerequisites do not prove network availability or upstream stability; the fetch command verifies the live page, pinned caption hash, and media duration.",
        ],
    }
    return report
