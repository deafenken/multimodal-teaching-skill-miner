from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import ensure_private_directory, ensure_private_file


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

EVENT_STRATEGY_MAP: dict[str, tuple[str, ...]] = {
    "question_and_wait": ("question_and_wait",),
    "scene_change": (),
    "slide_change": (),
    "board_build_up": (),
    "code_or_formula_visible": (),
    "code_formula_walkthrough": ("line_by_line_explanation", "step_by_step"),
    "visual_example": ("concrete_example",),
    "student_confusion": (),
    "teacher_adjustment": ("adaptive_teaching",),
    "student_answer": (),
}

QUESTION_MARKERS = ("?", "why", "what happens", "can you", "predict", "为什么", "会怎样", "你认为", "请判断")
VISUAL_EXAMPLE_MARKERS = ("example", "picture", "diagram", "draw", "例如", "图", "观察")
CODE_SPEECH_MARKERS = ("code", "line", "formula", "equation", "run", "代码", "逐行", "公式", "运行")
CODE_OCR_RE = re.compile(r"\b(def|for|while|if|else|return|class|import)\b|[={}()\[\];]|\b[A-Za-z]\s*=", re.IGNORECASE)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(command: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    version_flag = "--version" if "tesseract" in Path(executable).name.lower() else "-version"
    result = subprocess.run([executable, version_flag], capture_output=True, text=True, check=False)
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0] if first_line else executable


def probe_media(path: str | Path) -> dict[str, Any]:
    ffprobe = os.getenv("TSM_FFPROBE", "ffprobe")
    if not shutil.which(ffprobe):
        raise RuntimeError(f"multimodal analysis requires `{ffprobe}` on PATH")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    media_format = payload.get("format", {})
    return {
        "duration_seconds": round(float(media_format.get("duration", 0) or 0), 3),
        "size_bytes": int(media_format.get("size", 0) or 0),
        "bit_rate": int(media_format.get("bit_rate", 0) or 0),
        "streams": payload.get("streams", []),
    }


def parse_silencedetect(log_text: str, media_duration: float | None = None) -> list[dict[str, float]]:
    starts = re.findall(r"silence_start:\s*(-?\d+(?:\.\d+)?)", log_text)
    ends = re.findall(
        r"silence_end:\s*(-?\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*(\d+(?:\.\d+)?)",
        log_text,
    )
    events = [
        {"start": max(0.0, float(end) - float(duration)), "end": float(end), "duration": float(duration)}
        for end, duration in ends
    ]
    if len(starts) > len(ends) and media_duration is not None:
        start = max(0.0, float(starts[-1]))
        if media_duration > start:
            events.append({"start": start, "end": media_duration, "duration": media_duration - start})
    events.sort(key=lambda item: item["start"])
    return [{key: round(value, 3) for key, value in item.items()} for item in events]


def detect_silences(
    path: str | Path,
    *,
    media_duration: float,
    noise_db: float = -35.0,
    minimum_duration: float = 0.8,
) -> list[dict[str, float]]:
    ffmpeg = os.getenv("TSM_FFMPEG", "ffmpeg")
    if not shutil.which(ffmpeg):
        raise RuntimeError(f"multimodal analysis requires `{ffmpeg}` on PATH")
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={minimum_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr[-1000:]
        raise RuntimeError(f"ffmpeg silence detection failed: {detail}")
    return parse_silencedetect(result.stderr, media_duration)


def _extract_frames_with_filter(
    video_path: Path,
    frame_dir: Path,
    *,
    prefix: str,
    video_filter: str,
    maximum_frames: int,
) -> list[dict[str, Any]]:
    ffmpeg = os.getenv("TSM_FFMPEG", "ffmpeg")
    pattern = frame_dir / f"{prefix}_%04d.jpg"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"{video_filter},scale=960:-2,showinfo",
            "-frames:v",
            str(maximum_frames),
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            str(pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    timestamps = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
    paths = sorted(frame_dir.glob(f"{prefix}_*.jpg"))[: len(timestamps)]
    for frame_path in paths:
        ensure_private_file(frame_path)
    return [
        {
            "timestamp": round(timestamp, 3),
            "path": str(Path("frames") / path.name),
            "absolute_path": str(path),
            "sampling_source": prefix,
        }
        for path, timestamp in zip(paths, timestamps)
    ]


def _available_tesseract_language(requested: str) -> str | None:
    tesseract = os.getenv("TSM_TESSERACT", "tesseract")
    if not shutil.which(tesseract):
        return None
    result = subprocess.run([tesseract, "--list-langs"], capture_output=True, text=True, check=False)
    available = set(result.stdout.splitlines()[1:])
    candidates = ["chi_sim+eng", "chi_sim", "eng"] if requested.lower().startswith(("zh", "chi")) else ["eng"]
    for candidate in candidates:
        parts = candidate.split("+")
        if all(part in available for part in parts):
            return candidate
    return None


def _ocr_frame(path: str, language: str | None) -> str:
    tesseract = os.getenv("TSM_TESSERACT", "tesseract")
    if language is None or not shutil.which(tesseract):
        return ""
    result = subprocess.run(
        [tesseract, path, "stdout", "-l", language, "--psm", "6"],
        capture_output=True,
        text=True,
        check=False,
    )
    return re.sub(r"\s+", " ", result.stdout).strip()[:1500] if result.returncode == 0 else ""


def extract_keyframes(
    path: str | Path,
    output_dir: str | Path,
    *,
    language: str = "en",
    interval_seconds: float = 30.0,
    scene_threshold: float = 0.32,
    maximum_frames: int = 48,
    use_ocr: bool = True,
) -> list[dict[str, Any]]:
    ffmpeg = os.getenv("TSM_FFMPEG", "ffmpeg")
    if not shutil.which(ffmpeg):
        raise RuntimeError(f"multimodal analysis requires `{ffmpeg}` on PATH")
    frame_dir = ensure_private_directory(Path(output_dir) / "frames")
    for prefix in ("scene_", "uniform_"):
        for stale_frame in frame_dir.glob(f"{prefix}*.jpg"):
            stale_frame.unlink()
    video_path = Path(path)
    scene_frames = _extract_frames_with_filter(
        video_path,
        frame_dir,
        prefix="scene",
        video_filter=f"select='eq(n\\,0)+gt(scene\\,{scene_threshold})'",
        maximum_frames=max(1, maximum_frames // 2),
    )
    uniform_frames = _extract_frames_with_filter(
        video_path,
        frame_dir,
        prefix="uniform",
        video_filter=f"fps=1/{max(1.0, interval_seconds)}",
        maximum_frames=max(1, maximum_frames // 2),
    )
    merged: list[dict[str, Any]] = []
    for frame in sorted(scene_frames + uniform_frames, key=lambda item: (item["timestamp"], item["sampling_source"])):
        if any(abs(frame["timestamp"] - existing["timestamp"]) < 0.4 for existing in merged):
            if frame["sampling_source"] == "scene":
                nearest = min(merged, key=lambda item: abs(frame["timestamp"] - item["timestamp"]))
                nearest["sampling_source"] = "scene+uniform"
            continue
        merged.append(frame)
    ocr_language = _available_tesseract_language(language) if use_ocr else None
    for frame in merged:
        frame["ocr_text"] = _ocr_frame(frame.pop("absolute_path"), ocr_language)
        frame["ocr_language"] = ocr_language
    return merged[:maximum_frames]


def infer_visual_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    sorted_frames = sorted(frames, key=lambda item: item["timestamp"])
    for index, frame in enumerate(sorted_frames):
        timestamp = float(frame["timestamp"])
        text = str(frame.get("ocr_text", ""))
        if "scene" in str(frame.get("sampling_source", "")) and index > 0:
            events.append(
                {
                    "type": "scene_change",
                    "start": timestamp,
                    "end": timestamp,
                    "modalities": ["visual"],
                    "evidence": {"frame_path": frame["path"], "ocr_text": text[:240]},
                    "confidence": 0.75,
                }
            )
        if text and CODE_OCR_RE.search(text):
            events.append(
                {
                    "type": "code_or_formula_visible",
                    "start": timestamp,
                    "end": timestamp,
                    "modalities": ["visual", "ocr"],
                    "evidence": {"frame_path": frame["path"], "ocr_text": text[:240]},
                    "confidence": 0.72,
                }
            )
        if index == 0:
            continue
        previous = sorted_frames[index - 1]
        previous_text = str(previous.get("ocr_text", ""))
        if not text or not previous_text:
            continue
        similarity = difflib.SequenceMatcher(None, previous_text.lower(), text.lower()).ratio()
        previous_tokens = previous_text.split()
        tokens = text.split()
        if similarity < 0.35:
            events.append(
                {
                    "type": "slide_change",
                    "start": float(previous["timestamp"]),
                    "end": timestamp,
                    "modalities": ["visual", "ocr"],
                    "evidence": {
                        "before_frame": previous["path"],
                        "after_frame": frame["path"],
                        "before_text": previous_text[:180],
                        "after_text": text[:180],
                        "text_similarity": round(similarity, 3),
                    },
                    "confidence": round(0.65 + min(0.25, (0.35 - similarity)), 2),
                }
            )
        elif similarity >= 0.45 and len(tokens) >= max(3, int(len(previous_tokens) * 1.25)):
            events.append(
                {
                    "type": "board_build_up",
                    "start": float(previous["timestamp"]),
                    "end": timestamp,
                    "modalities": ["visual", "ocr"],
                    "evidence": {
                        "before_frame": previous["path"],
                        "after_frame": frame["path"],
                        "added_token_count": len(tokens) - len(previous_tokens),
                        "after_text": text[:240],
                    },
                    "confidence": 0.76,
                }
            )
    return events


def _near_segment(event: dict[str, Any], segments: list[dict[str, Any]], window: float = 12.0) -> dict[str, Any] | None:
    event_time = (float(event["start"]) + float(event["end"])) / 2
    matches = [
        segment
        for segment in segments
        if float(segment["start"]) - window <= event_time <= float(segment["end"]) + window
    ]
    return min(matches, key=lambda segment: abs(((segment["start"] + segment["end"]) / 2) - event_time)) if matches else None


def fuse_multimodal_events(
    transcript: dict[str, Any],
    silences: list[dict[str, float]],
    visual_events: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
    *,
    language_modality: str = "speech",
) -> list[dict[str, Any]]:
    if language_modality not in {"speech", "transcript"}:
        raise ValueError("language_modality must be speech or transcript")
    segments = transcript.get("segments", [])
    events: list[dict[str, Any]] = []
    for segment in segments:
        lowered = str(segment.get("text", "")).lower()
        if not any(marker in lowered for marker in QUESTION_MARKERS):
            continue
        candidates = [
            silence
            for silence in silences
            if silence["end"] > float(segment["end"])
            and silence["start"] <= float(segment["end"]) + 3.0
            and silence["end"] - max(silence["start"], float(segment["end"])) >= 0.5
        ]
        if candidates:
            silence = min(candidates, key=lambda item: abs(item["start"] - float(segment["end"])))
            wait_start = max(silence["start"], float(segment["end"]))
            wait_seconds = silence["end"] - wait_start
            events.append(
                {
                    "type": "question_and_wait",
                    "start": float(segment["start"]),
                    "end": silence["end"],
                    "modalities": [language_modality, "audio"],
                    "evidence": {
                        "speech_quote": segment["text"],
                        "question_segment": {"start": segment["start"], "end": segment["end"]},
                        "silence": silence,
                        "wait_interval": {"start": round(wait_start, 3), "end": silence["end"]},
                        "wait_seconds": round(wait_seconds, 3),
                    },
                    "confidence": round(min(0.98, 0.68 + wait_seconds / 12), 2),
                }
            )
    events.extend(visual_events)
    for event in visual_events:
        nearby = _near_segment(event, segments)
        if not nearby:
            continue
        lowered = str(nearby["text"]).lower()
        if event["type"] == "code_or_formula_visible" and any(marker in lowered for marker in CODE_SPEECH_MARKERS):
            events.append(
                {
                    "type": "code_formula_walkthrough",
                    "start": min(float(event["start"]), float(nearby["start"])),
                    "end": max(float(event["end"]), float(nearby["end"])),
                    "modalities": [language_modality, "visual", "ocr"],
                    "evidence": {"speech_quote": nearby["text"], **event["evidence"]},
                    "confidence": 0.86,
                }
            )
        if event["type"] in {"scene_change", "slide_change", "board_build_up"} and any(
            marker in lowered for marker in VISUAL_EXAMPLE_MARKERS
        ):
            events.append(
                {
                    "type": "visual_example",
                    "start": min(float(event["start"]), float(nearby["start"])),
                    "end": max(float(event["end"]), float(nearby["end"])),
                    "modalities": [language_modality, "visual"],
                    "evidence": {"speech_quote": nearby["text"], **event["evidence"]},
                    "confidence": 0.82,
                }
            )
    for observation in observations or []:
        identity_keys = {
            "student",
            "student_name",
            "student_id",
            "name",
            "face_id",
            "identity",
            "person_id",
            "email",
            "phone",
        }
        if any(str(key).lower() in identity_keys for key in observation):
            raise ValueError("classroom observations must be anonymized; identity fields are not allowed")
        event_type = str(observation.get("type", ""))
        if event_type not in {"student_confusion", "teacher_adjustment", "student_answer"}:
            raise ValueError(f"unsupported classroom observation type: {event_type}")
        note = str(observation.get("note", "")).strip()
        if not note:
            raise ValueError("classroom observations require a non-empty anonymized note")
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:\+?\d[\d\s-]{7,}\d)", note):
            raise ValueError("classroom observation note appears to contain contact information")
        confidence = float(observation.get("confidence", 0.8))
        if not 0 <= confidence <= 1:
            raise ValueError("classroom observation confidence must be between 0 and 1")
        events.append(
            {
                "type": event_type,
                "start": float(observation["start"]),
                "end": float(observation.get("end", observation["start"])),
                "modalities": ["classroom_observation"],
                "evidence": {
                    "anonymized_note": note[:300],
                    "evidence_origin": "provided_anonymized_annotation",
                },
                "confidence": confidence,
            }
        )
    events.sort(key=lambda item: (float(item["start"]), item["type"]))
    for index, event in enumerate(events, 1):
        event["event_id"] = f"mme_{index:04d}"
        event["supports_strategies"] = list(EVENT_STRATEGY_MAP.get(event["type"], ()))
    return events


def analyze_video(
    video_path: str | Path,
    transcript: dict[str, Any],
    output_dir: str | Path,
    *,
    observations: list[dict[str, Any]] | None = None,
    use_ocr: bool = True,
    frame_interval_seconds: float = 30.0,
    maximum_frames: int = 48,
) -> dict[str, Any]:
    if frame_interval_seconds <= 0:
        raise ValueError("frame_interval_seconds must be positive")
    if maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"multimodal visual analysis requires a video file, got {path.suffix}")
    output = ensure_private_directory(output_dir)
    media = probe_media(path)
    media["sha256"] = _file_sha256(path)
    streams = media.get("streams", [])
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    silences = detect_silences(path, media_duration=media["duration_seconds"]) if has_audio else []
    frames = (
        extract_keyframes(
            path,
            output,
            language=str(transcript.get("language", "en")),
            interval_seconds=frame_interval_seconds,
            maximum_frames=maximum_frames,
            use_ocr=use_ocr,
        )
        if has_video
        else []
    )
    visual_events = infer_visual_events(frames)
    audio_content_verified = bool(
        transcript.get("transcript_kind") == "automatic_speech_recognition"
        and transcript.get("provenance", {}).get("input_sha256") == media["sha256"]
    )
    language_modality = "speech" if audio_content_verified else "transcript"
    events = fuse_multimodal_events(
        transcript,
        silences,
        visual_events,
        observations,
        language_modality=language_modality,
    )
    total_silence = sum(item["duration"] for item in silences)
    modalities_available = [language_modality]
    if has_audio:
        modalities_available.append("audio")
    if has_video:
        modalities_available.append("visual")
    if any(frame.get("ocr_text") for frame in frames):
        modalities_available.append("ocr")
    if observations:
        modalities_available.append("classroom_observation")
    return {
        "schema_version": "1.0",
        "media": media,
        "language": {
            "modality": language_modality,
            "status": "asr_verified_against_media" if audio_content_verified else "provided_transcript",
            "transcript_kind": transcript.get("transcript_kind"),
            "audio_content_verified": audio_content_verified,
            "note": "A supplied transcript is a text modality; speech/audio correspondence is only verified when ASR provenance hashes the same media.",
        },
        "modalities_available": modalities_available,
        "audio": {
            "status": "available" if has_audio else "unavailable",
            "silences": silences,
            "silence_ratio": round(total_silence / media["duration_seconds"], 4) if media["duration_seconds"] else 0.0,
        },
        "visual": {
            "status": "available" if has_video else "unavailable",
            "keyframes": frames,
            "events": visual_events,
            "ocr_enabled": use_ocr,
        },
        "events": events,
        "privacy": {
            "identity_recognition_performed": False,
            "classroom_observations_must_be_anonymized": True,
        },
        "provenance": {
            "pipeline": "teaching_skill_miner.multimodal.v1",
            "ffmpeg": _tool_version(os.getenv("TSM_FFMPEG", "ffmpeg")),
            "ffprobe": _tool_version(os.getenv("TSM_FFPROBE", "ffprobe")),
            "tesseract": _tool_version(os.getenv("TSM_TESSERACT", "tesseract")) if use_ocr else None,
            "frame_interval_seconds": frame_interval_seconds,
            "maximum_frames": maximum_frames,
        },
    }


def enrich_transcript_with_multimodal(
    transcript: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(transcript)
    enriched["multimodal"] = analysis
    return enriched
