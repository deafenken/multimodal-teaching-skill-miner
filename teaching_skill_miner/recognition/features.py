from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("real recognition requires the optional `recognition` dependencies") from exc
    return np


def _fixed_window(
    duration: float,
    maximum_seconds: float,
    start_time: float = 0.0,
) -> tuple[float, float]:
    if duration <= maximum_seconds:
        return start_time, max(0.2, duration)
    return start_time + (duration - maximum_seconds) / 2.0, maximum_seconds


def _extract_frames(
    path: Path,
    *,
    duration: float,
    start_time: float,
    frame_count: int,
    width: int,
    height: int,
    clip_seconds: float,
) -> Any:
    np = _numpy()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("real classroom recognition requires ffmpeg on PATH")
    start, window = _fixed_window(duration, clip_seconds, start_time)
    fps = max(0.01, frame_count / window)
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{window:.6f}",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps:.8f},scale={width}:{height}:flags=area,format=rgb24",
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    expected = width * height * 3
    count = len(result.stdout) // expected
    if result.returncode != 0 or count == 0:
        raise RuntimeError(f"frame extraction failed for {path.name}: {result.stderr.decode(errors='ignore')[-500:]}")
    frames = np.frombuffer(result.stdout[: count * expected], dtype=np.uint8).reshape(count, height, width, 3)
    if count < frame_count:
        frames = np.concatenate([frames, np.repeat(frames[-1:], frame_count - count, axis=0)], axis=0)
    return frames.astype(np.float32) / 255.0


def _extract_audio(
    path: Path,
    *,
    duration: float,
    start_time: float,
    clip_seconds: float,
    sample_rate: int,
    audio_stream_index: int | None,
) -> Any:
    np = _numpy()
    if audio_stream_index is None:
        return np.zeros(0, dtype=np.float32)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("real classroom recognition requires ffmpeg on PATH")
    start, window = _fixed_window(duration, clip_seconds, start_time)
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{window:.6f}",
            "-i",
            str(path),
            "-map",
            "0:a:0?" if audio_stream_index < 0 else f"0:{audio_stream_index}?",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if not result.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _pool_image(image: Any, rows: int = 9, columns: int = 16) -> Any:
    height, width = image.shape
    usable_h = rows * (height // rows)
    usable_w = columns * (width // columns)
    cropped = image[:usable_h, :usable_w]
    return cropped.reshape(rows, usable_h // rows, columns, usable_w // columns).mean(axis=(1, 3))


def visual_features(frames: Any) -> tuple[Any, list[str]]:
    np = _numpy()
    gray = frames @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mean_gray = gray.mean(axis=0)
    std_gray = gray.std(axis=0)
    pooled_mean = _pool_image(mean_gray).reshape(-1)
    pooled_std = _pool_image(std_gray).reshape(-1)
    values: list[float] = []
    names: list[str] = []
    for channel, channel_name in enumerate(("r", "g", "b")):
        channel_values = frames[..., channel]
        for statistic, value in (("mean", channel_values.mean()), ("std", channel_values.std())):
            values.append(float(value))
            names.append(f"visual_{channel_name}_{statistic}")
        histogram, _ = np.histogram(channel_values, bins=8, range=(0.0, 1.0), density=True)
        values.extend(float(item) for item in histogram)
        names.extend(f"visual_{channel_name}_hist_{index}" for index in range(8))
    values.extend(float(item) for item in pooled_mean)
    names.extend(f"visual_mean_cell_{index}" for index in range(len(pooled_mean)))
    values.extend(float(item) for item in pooled_std)
    names.extend(f"visual_temporal_std_cell_{index}" for index in range(len(pooled_std)))

    horizontal = np.abs(np.diff(gray, axis=2))
    vertical = np.abs(np.diff(gray, axis=1))
    motion = np.abs(np.diff(gray, axis=0)) if len(gray) > 1 else np.zeros_like(gray[:1])
    for feature_name, array in (("edge_x", horizontal), ("edge_y", vertical), ("motion", motion)):
        for statistic, value in (
            ("mean", array.mean()),
            ("std", array.std()),
            ("p90", np.quantile(array, 0.9)),
        ):
            values.append(float(value))
            names.append(f"visual_{feature_name}_{statistic}")
    motion_map = motion.mean(axis=0)
    pooled_motion = _pool_image(motion_map).reshape(-1)
    values.extend(float(item) for item in pooled_motion)
    names.extend(f"visual_motion_cell_{index}" for index in range(len(pooled_motion)))
    return np.asarray(values, dtype=np.float32), names


def audio_features(samples: Any, sample_rate: int) -> tuple[Any, list[str]]:
    np = _numpy()
    statistic_names = ("mean", "std", "p25", "p50", "p75")
    descriptor_names = (
        "log_energy",
        "zero_crossing",
        "spectral_centroid",
        "spectral_flatness",
        "band_0_300",
        "band_300_1000",
        "band_1000_3000",
        "band_3000_4000",
    )
    names = [f"audio_{descriptor}_{statistic}" for descriptor in descriptor_names for statistic in statistic_names]
    names.extend(["audio_silence_ratio", "audio_present"])
    if len(samples) < 256:
        return np.zeros(len(names), dtype=np.float32), names
    frame_size = 256
    count = len(samples) // frame_size
    framed = samples[: count * frame_size].reshape(count, frame_size)
    windowed = framed * np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-8
    power = spectrum**2
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    log_energy = np.log10(np.mean(framed**2, axis=1) + 1e-10)
    zero_crossing = np.mean(np.signbit(framed[:, 1:]) != np.signbit(framed[:, :-1]), axis=1)
    spectral_centroid = (power * frequencies).sum(axis=1) / power.sum(axis=1) / (sample_rate / 2)
    spectral_flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    descriptors = [log_energy, zero_crossing, spectral_centroid, spectral_flatness]
    for low, high in ((0, 300), (300, 1000), (1000, 3000), (3000, 4000)):
        mask = (frequencies >= low) & (frequencies < high)
        descriptors.append(power[:, mask].sum(axis=1) / power.sum(axis=1))
    values: list[float] = []
    for descriptor in descriptors:
        values.extend(
            [
                float(descriptor.mean()),
                float(descriptor.std()),
                float(np.quantile(descriptor, 0.25)),
                float(np.quantile(descriptor, 0.5)),
                float(np.quantile(descriptor, 0.75)),
            ]
        )
    values.append(float(np.mean(log_energy < -4.0)))
    values.append(1.0)
    return np.asarray(values, dtype=np.float32), names


def extract_video_visual_features(
    path: str | Path,
    *,
    duration: float,
    start_time: float = 0.0,
    frame_count: int = 12,
    width: int = 64,
    height: int = 36,
    clip_seconds: float = 10.0,
) -> dict[str, Any]:
    """Extract the fixed handcrafted visual descriptor from one media window."""

    frames = _extract_frames(
        Path(path),
        duration=duration,
        start_time=start_time,
        frame_count=frame_count,
        width=width,
        height=height,
        clip_seconds=clip_seconds,
    )
    features, feature_names = visual_features(frames)
    return {
        "features": features,
        "feature_names": feature_names,
        "configuration": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "clip_seconds": clip_seconds,
            "window_policy": "center_crop_or_full_if_shorter",
            "video_stream_start_time_seconds": start_time,
        },
    }


def extract_video_audio_features(
    path: str | Path,
    *,
    duration: float,
    start_time: float = 0.0,
    clip_seconds: float = 10.0,
    sample_rate: int = 8000,
    audio_stream_index: int | None = -1,
) -> dict[str, Any]:
    """Extract the fixed handcrafted audio descriptor from one media window."""

    samples = _extract_audio(
        Path(path),
        duration=duration,
        start_time=start_time,
        clip_seconds=clip_seconds,
        sample_rate=sample_rate,
        audio_stream_index=audio_stream_index,
    )
    features, feature_names = audio_features(samples, sample_rate)
    return {
        "features": features,
        "feature_names": feature_names,
        "audio_present": bool(len(samples)),
        "configuration": {
            "clip_seconds": clip_seconds,
            "sample_rate": sample_rate,
            "window_policy": "center_crop_or_full_if_shorter",
            "video_stream_start_time_seconds": start_time,
            "selected_audio_stream_index": audio_stream_index,
        },
    }


def extract_multimodal_features(
    path: str | Path,
    *,
    duration: float,
    start_time: float = 0.0,
    frame_count: int = 12,
    width: int = 64,
    height: int = 36,
    clip_seconds: float = 10.0,
    sample_rate: int = 8000,
    audio_stream_index: int | None = -1,
) -> dict[str, Any]:
    path = Path(path)
    visual_result = extract_video_visual_features(
        path,
        duration=duration,
        start_time=start_time,
        frame_count=frame_count,
        width=width,
        height=height,
        clip_seconds=clip_seconds,
    )
    audio_result = extract_video_audio_features(
        path,
        duration=duration,
        start_time=start_time,
        clip_seconds=clip_seconds,
        sample_rate=sample_rate,
        audio_stream_index=audio_stream_index,
    )
    visual = visual_result["features"]
    visual_names = visual_result["feature_names"]
    audio = audio_result["features"]
    audio_names = audio_result["feature_names"]
    np = _numpy()
    return {
        "features": np.concatenate([visual, audio]),
        "feature_names": visual_names + audio_names,
        "visual_dimension": len(visual),
        "audio_dimension": len(audio),
        "audio_present": audio_result["audio_present"],
        "configuration": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "clip_seconds": clip_seconds,
            "sample_rate": sample_rate,
            "window_policy": "center_crop_or_full_if_shorter",
            "video_stream_start_time_seconds": start_time,
            "selected_audio_stream_index": audio_stream_index,
        },
    }
