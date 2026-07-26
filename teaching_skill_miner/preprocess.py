from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .models import validate_transcript


TIMESTAMP_RE = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{2}):(?P<s>\d{2})(?P<frac>[,.]\d{1,3})?"
)
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".mp3", ".wav", ".m4a"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0].strip() if lines else None


def _whisper_version(command: str) -> str | None:
    try:
        return importlib.metadata.version("openai-whisper")
    except importlib.metadata.PackageNotFoundError:
        return _command_version(command)


def _whisper_model_sha256(model: str) -> str | None:
    declared = os.getenv("TSM_WHISPER_MODEL_SHA256", "").strip().lower()
    if declared:
        if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
            raise ValueError("TSM_WHISPER_MODEL_SHA256 must be a lowercase SHA-256 hex digest")
        return declared
    configured_path = os.getenv("TSM_WHISPER_MODEL_PATH", "").strip()
    candidates = [
        Path(configured_path).expanduser() if configured_path else None,
        Path.home() / ".cache" / "whisper" / f"{model}.pt",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return _file_sha256(candidate)
    return None


def _media_duration_seconds(path: Path) -> float | None:
    ffprobe = os.getenv("TSM_FFPROBE", "ffprobe")
    if not shutil.which(ffprobe):
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        value = float(result.stdout.strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return value if result.returncode == 0 and value > 0 else None


def timestamp_seconds(value: str) -> float:
    match = TIMESTAMP_RE.search(value.strip())
    if not match:
        raise ValueError(f"invalid timestamp: {value}")
    fraction = (match.group("frac") or "").replace(",", ".")
    return (
        int(match.group("h") or 0) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + (float(fraction) if fraction else 0.0)
    )


def parse_srt_or_vtt(text: str) -> list[dict[str, Any]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        left, right = lines[timing_index].split("-->", 1)
        right = right.split()[0]
        clean_text = " ".join(lines[timing_index + 1 :])
        clean_text = re.sub(r"<[^>]+>", "", clean_text).strip()
        if not clean_text:
            continue
        segments.append(
            {"start": timestamp_seconds(left), "end": timestamp_seconds(right), "text": clean_text}
        )
    return merge_duplicate_captions(segments)


def merge_duplicate_captions(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in segments:
        if merged and segment["text"] == merged[-1]["text"]:
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
        else:
            merged.append(dict(segment))
    return merged


def chunk_plain_text(text: str, seconds_per_chunk: int = 30) -> list[dict[str, Any]]:
    if seconds_per_chunk < 1:
        raise ValueError("seconds_per_chunk must be positive")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if part.strip()]
    if not sentences:
        return []
    segments: list[dict[str, Any]] = []
    for index in range(0, len(sentences), 3):
        start = (index // 3) * seconds_per_chunk
        segments.append(
            {
                "start": start,
                "end": start + seconds_per_chunk,
                "text": " ".join(sentences[index : index + 3]),
            }
        )
    return segments


def transcribe_media(path: Path, language: str) -> list[dict[str, Any]]:
    ffmpeg = os.getenv("TSM_FFMPEG", "ffmpeg")
    whisper = os.getenv("TSM_WHISPER", "whisper")
    if not shutil.which(ffmpeg):
        raise RuntimeError(f"media input requires `{ffmpeg}` on PATH")
    if not shutil.which(whisper):
        raise RuntimeError(f"media input requires `{whisper}` on PATH")
    timeout = int(os.getenv("TSM_MEDIA_TIMEOUT_SECONDS", "1800"))
    if timeout < 1:
        raise ValueError("TSM_MEDIA_TIMEOUT_SECONDS must be positive")
    whisper_model = os.getenv("TSM_WHISPER_MODEL", "base").strip() or "base"
    whisper_device = os.getenv("TSM_WHISPER_DEVICE", "").strip()
    with tempfile.TemporaryDirectory(prefix="tsm-") as temp_dir:
        audio = Path(temp_dir) / "audio.wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", "16000", str(audio)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        whisper_command = [
            whisper,
            str(audio),
            "--language",
            language,
            "--model",
            whisper_model,
            "--output_format",
            "json",
            "--output_dir",
            temp_dir,
        ]
        if whisper_device:
            whisper_command.extend(["--device", whisper_device])
        subprocess.run(
            whisper_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = json.loads((Path(temp_dir) / "audio.json").read_text(encoding="utf-8"))
        return [
            {"start": float(item["start"]), "end": float(item["end"]), "text": item["text"].strip()}
            for item in result.get("segments", [])
            if item.get("text", "").strip()
        ]


def preprocess_file(
    input_path: str | Path,
    *,
    video_id: str,
    course_id: str,
    title: str,
    source_url: str,
    language: str = "en",
) -> dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".srt", ".vtt"}:
        segments = parse_srt_or_vtt(path.read_text(encoding="utf-8-sig"))
        transcript_kind = "caption_import"
    elif suffix == ".txt":
        segments = chunk_plain_text(path.read_text(encoding="utf-8-sig"))
        transcript_kind = "plain_text_import"
    elif suffix in MEDIA_EXTENSIONS:
        segments = transcribe_media(path, language)
        transcript_kind = "automatic_speech_recognition"
    else:
        raise ValueError(f"unsupported input type: {suffix}")
    transcript = {
        "video_id": video_id,
        "course_id": course_id,
        "title": title,
        "source_url": source_url,
        "language": language,
        "transcript_kind": transcript_kind,
        "timestamps_are_approximate": transcript_kind == "plain_text_import",
        "provenance": {
            "input_filename": path.name,
            "input_sha256": _file_sha256(path),
            "input_size_bytes": path.stat().st_size,
            "pipeline": "teaching_skill_miner.preprocess.v1",
            "timestamp_source": "source_caption" if transcript_kind == "caption_import" else "asr" if transcript_kind == "automatic_speech_recognition" else "synthetic_chunks",
        },
        "segments": segments,
    }
    if transcript_kind == "automatic_speech_recognition":
        whisper_model = os.getenv("TSM_WHISPER_MODEL", "base").strip() or "base"
        whisper_command = os.getenv("TSM_WHISPER", "whisper")
        source_duration = _media_duration_seconds(path)
        covered_duration = max(
            (float(segment["end"]) for segment in segments), default=0.0
        )
        decoding_config = {
            "engine": "openai_whisper_cli",
            "command": whisper_command,
            "model": whisper_model,
            "device": os.getenv("TSM_WHISPER_DEVICE", "auto") or "auto",
            "language": language,
            "audio_channels": 1,
            "audio_sample_rate_hz": 16000,
            "timeout_seconds": int(os.getenv("TSM_MEDIA_TIMEOUT_SECONDS", "1800")),
        }
        transcript["provenance"]["asr"] = {
            **decoding_config,
            "version": _whisper_version(whisper_command),
            "model_sha256": _whisper_model_sha256(whisper_model),
            "decoding_config_sha256": hashlib.sha256(
                json.dumps(
                    decoding_config,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        transcript["provenance"]["transcript_coverage"] = {
            "source_duration_seconds": source_duration,
            "covered_duration_seconds": covered_duration,
            "coverage_fraction": (
                min(1.0, covered_duration / source_duration)
                if source_duration
                else None
            ),
            "completeness_verified": bool(
                source_duration and covered_duration >= 0.95 * source_duration
            ),
            "verification_method": "full_media_openai_whisper_cli_run",
        }
    result = validate_transcript(transcript)
    if not result.valid:
        raise ValueError("; ".join(result.errors))
    return transcript
