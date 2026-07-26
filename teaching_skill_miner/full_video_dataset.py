from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable
from urllib.parse import unquote, urlsplit, urlunsplit

from .formal_captions import MIT_OCW_CAPTION_LICENSE_URL
from .io_utils import (
    ensure_private_directory,
    ensure_private_file,
    read_json,
    write_json,
)


ARCHIVE_CANONICAL_HOST = "archive.org"
ARCHIVE_SOURCE_HOSTS = {"archive.org", "www.archive.org"}
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_CONTAINER_NAMES = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_json_object(path: str | Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def _safe_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    text = value.strip()
    if "\\" in text or "\x00" in text:
        raise ValueError(f"{field} is unsafe")
    path = Path(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} is unsafe")
    return path


def _url_host_is_archive(hostname: str | None, *, allow_cdn: bool) -> bool:
    host = (hostname or "").lower().rstrip(".")
    if host in ARCHIVE_SOURCE_HOSTS:
        return True
    return bool(allow_cdn and host.endswith(".archive.org"))


def require_archive_https_url(value: Any, *, allow_cdn: bool = False) -> str:
    """Validate an Internet Archive HTTPS URL without normalizing its host."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("media URL must be non-empty")
    url = value.strip()
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("media URL has an invalid port") from exc
    if (
        parsed.scheme != "https"
        or not _url_host_is_archive(parsed.hostname, allow_cdn=allow_cdn)
        or parsed.username
        or parsed.password
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise ValueError("media URL must use an approved archive.org HTTPS host")
    decoded_path = unquote(parsed.path)
    if (
        not parsed.path.startswith("/")
        or "\\" in decoded_path
        or any(character.isspace() or ord(character) < 32 for character in decoded_path)
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    ):
        raise ValueError("media URL path is unsafe")
    return url


def normalize_archive_media_url(value: Any) -> str:
    """Normalize an OCW-declared Archive URL to canonical HTTPS.

    Source manifests may contain the historic ``http://www.archive.org`` form.
    Only exact first-party source hosts are accepted here; CDN subdomains are
    accepted only when validating curl's final effective URL.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("media URL must be non-empty")
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("media URL has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not _url_host_is_archive(parsed.hostname, allow_cdn=False)
        or parsed.username
        or parsed.password
        or port not in {None, 80, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "source media URL must use archive.org without credentials, query, "
            "or fragment"
        )
    decoded_path = unquote(parsed.path)
    if (
        not parsed.path.startswith("/download/")
        or "\\" in decoded_path
        or any(character.isspace() or ord(character) < 32 for character in decoded_path)
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    ):
        raise ValueError("source media URL must use a safe /download/ path")
    return urlunsplit(("https", ARCHIVE_CANONICAL_HOST, parsed.path, "", ""))


def build_download_plan(
    source_manifest: dict[str, Any],
    formal_caption_manifest: dict[str, Any],
    *,
    source_manifest_sha256: str,
    expected_video_count: int = 10,
) -> list[dict[str, Any]]:
    """Purely validate and bind the source and formal-caption manifests."""

    if source_manifest.get("source_kind") != "mit_ocw_official_caption_index":
        raise ValueError("unsupported source manifest kind")
    if source_manifest.get("publisher") != "MIT OpenCourseWare":
        raise ValueError("source publisher must be MIT OpenCourseWare")
    if source_manifest.get("license_url") != MIT_OCW_CAPTION_LICENSE_URL:
        raise ValueError("source manifest license is not the verified MIT OCW license")
    if not SHA256_RE.fullmatch(str(source_manifest_sha256).lower()):
        raise ValueError("source_manifest_sha256 must be a SHA-256 digest")
    if formal_caption_manifest.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("formal caption manifest is not bound to this source manifest")
    if formal_caption_manifest.get("dataset_id") != source_manifest.get("dataset_id"):
        raise ValueError("formal caption and source dataset IDs differ")
    if formal_caption_manifest.get("publisher") != source_manifest.get("publisher"):
        raise ValueError("formal caption and source publishers differ")
    if formal_caption_manifest.get("license_url") != source_manifest.get("license_url"):
        raise ValueError("formal caption and source licenses differ")

    source_videos = source_manifest.get("videos")
    formal_videos = formal_caption_manifest.get("videos")
    if not isinstance(source_videos, list) or not isinstance(formal_videos, list):
        raise ValueError("both manifests must contain video arrays")
    if expected_video_count < 1:
        raise ValueError("expected_video_count must be positive")
    if len(source_videos) != expected_video_count:
        raise ValueError(
            f"source manifest must contain exactly {expected_video_count} videos"
        )
    if len(formal_videos) != expected_video_count:
        raise ValueError(
            f"formal caption manifest must contain exactly {expected_video_count} videos"
        )

    formal_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(formal_videos):
        if not isinstance(item, dict):
            raise ValueError(f"formal videos[{index}] must be an object")
        video_id = item.get("video_id")
        if not isinstance(video_id, str) or not VIDEO_ID_RE.fullmatch(video_id):
            raise ValueError(f"formal videos[{index}].video_id is unsafe")
        if video_id in formal_by_id:
            raise ValueError(f"duplicate formal video_id: {video_id}")
        formal_by_id[video_id] = item

    plan: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for index, item in enumerate(source_videos):
        if not isinstance(item, dict):
            raise ValueError(f"source videos[{index}] must be an object")
        video_id = item.get("video_id")
        if not isinstance(video_id, str) or not VIDEO_ID_RE.fullmatch(video_id):
            raise ValueError(f"source videos[{index}].video_id is unsafe")
        if video_id in seen_source_ids:
            raise ValueError(f"duplicate source video_id: {video_id}")
        seen_source_ids.add(video_id)
        formal = formal_by_id.get(video_id)
        if formal is None:
            raise ValueError(f"formal caption manifest is missing {video_id}")
        for field in ("course_id", "title", "source_url"):
            source_value = item.get(field)
            if not isinstance(source_value, str) or not source_value.strip():
                raise ValueError(f"source {video_id}.{field} must be non-empty")
            if formal.get(field) != source_value:
                raise ValueError(f"formal caption identity mismatch for {video_id}.{field}")
        try:
            reference_duration = float(item.get("reference_media_duration_seconds"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid reference duration for {video_id}") from exc
        if not math.isfinite(reference_duration) or reference_duration <= 0:
            raise ValueError(f"invalid reference duration for {video_id}")
        caption_digest = str(item.get("caption_sha256", "")).lower()
        if not SHA256_RE.fullmatch(caption_digest):
            raise ValueError(f"invalid caption SHA-256 for {video_id}")
        transcript_path = _safe_relative_path(
            formal.get("transcript_path"),
            field=f"formal transcript_path for {video_id}",
        )
        plan.append(
            {
                "video_id": video_id,
                "course_id": item["course_id"],
                "title": item["title"],
                "source_url": item["source_url"],
                "media_url": normalize_archive_media_url(item.get("media_url")),
                "reference_media_duration_seconds": reference_duration,
                "caption_sha256": caption_digest,
                "formal_transcript_path": transcript_path.as_posix(),
            }
        )
    if set(formal_by_id) != seen_source_ids:
        extras = sorted(set(formal_by_id) - seen_source_ids)
        raise ValueError(f"formal caption manifest contains unexpected videos: {extras}")
    return plan


def load_full_video_plan(
    source_manifest_path: str | Path,
    formal_caption_manifest_path: str | Path,
    *,
    expected_video_count: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load manifests and bind each planned video to a verified transcript file."""

    source, source_raw = _read_json_object(
        source_manifest_path, label="source manifest"
    )
    formal_path = Path(formal_caption_manifest_path)
    formal, formal_raw = _read_json_object(
        formal_path, label="formal caption manifest"
    )
    source_digest = sha256(source_raw).hexdigest()
    plan = build_download_plan(
        source,
        formal,
        source_manifest_sha256=source_digest,
        expected_video_count=expected_video_count,
    )
    formal_root = formal_path.resolve().parent
    bound_plan: list[dict[str, Any]] = []
    for item in plan:
        relative_path = _safe_relative_path(
            item["formal_transcript_path"], field="formal_transcript_path"
        )
        transcript_path = formal_root / relative_path
        if transcript_path.is_symlink() or not transcript_path.is_file():
            raise ValueError(
                f"formal transcript is missing or is a symlink: {item['video_id']}"
            )
        transcript, transcript_raw = _read_json_object(
            transcript_path, label=f"formal transcript {item['video_id']}"
        )
        for field in ("video_id", "course_id", "title", "source_url"):
            if transcript.get(field) != item[field]:
                raise ValueError(
                    f"formal transcript identity mismatch for {item['video_id']}.{field}"
                )
        provenance = transcript.get("provenance")
        if not isinstance(provenance, dict) or provenance.get(
            "caption_source_verified"
        ) is not True:
            raise ValueError(
                f"formal transcript source was not verified for {item['video_id']}"
            )
        verification = provenance.get("caption_source_verification")
        if not isinstance(verification, dict):
            raise ValueError(
                f"formal transcript lacks source verification for {item['video_id']}"
            )
        transcript_media_url = normalize_archive_media_url(
            verification.get("media_probe_url")
        )
        if transcript_media_url != item["media_url"]:
            raise ValueError(
                f"formal transcript media URL mismatch for {item['video_id']}"
            )
        if str(provenance.get("input_sha256", "")).lower() != item["caption_sha256"]:
            raise ValueError(
                f"formal transcript caption hash mismatch for {item['video_id']}"
            )
        bound_item = dict(item)
        bound_item["formal_transcript_sha256"] = sha256(transcript_raw).hexdigest()
        bound_plan.append(bound_item)
    return (
        bound_plan,
        {
            "source_manifest_sha256": source_digest,
            "formal_caption_manifest_sha256": sha256(formal_raw).hexdigest(),
            "dataset_id": str(source.get("dataset_id", "")),
            "publisher": str(source.get("publisher", "")),
            "license_url": str(source.get("license_url", "")),
        },
    )


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_curl_write_out(stdout: str) -> dict[str, Any]:
    """Parse the three-line machine-only trailer emitted by curl."""

    lines = stdout.rstrip("\n").splitlines()
    if len(lines) != 3:
        raise RuntimeError("curl did not emit the expected download receipt")
    effective_url, status_text, downloaded_text = lines
    effective_url = require_archive_https_url(effective_url, allow_cdn=True)
    try:
        status_code = int(status_text)
        downloaded_bytes = int(round(float(downloaded_text)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("curl emitted an invalid download receipt") from exc
    if not 200 <= status_code < 300 or downloaded_bytes < 0:
        raise RuntimeError("curl emitted an unsuccessful download receipt")
    return {
        "effective_url": effective_url,
        "http_status": status_code,
        "downloaded_bytes_this_attempt": downloaded_bytes,
    }


def _resolve_executable(command: str, *, label: str) -> str:
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError(f"full-video download requires `{command}` on PATH")
    return executable


def curl_download_to_partial(
    media_url: str,
    partial_path: str | Path,
    *,
    curl_command: str = "curl",
    connect_timeout_seconds: int = 30,
    download_timeout_seconds: int = 14_400,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Resume one curl download into a private partial file."""

    url = normalize_archive_media_url(media_url)
    if connect_timeout_seconds < 1 or download_timeout_seconds < 1:
        raise ValueError("curl timeouts must be positive")
    executable = _resolve_executable(curl_command, label="curl")
    partial = Path(partial_path)
    ensure_private_directory(partial.parent)
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f"partial download target is unsafe: {partial}")
    if not partial.exists():
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
    ensure_private_file(partial)
    resumed_from_bytes = partial.stat().st_size
    command = [
        executable,
        "--location",
        "--max-redirs",
        "10",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--fail",
        "--show-error",
        "--silent",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        str(connect_timeout_seconds),
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--continue-at",
        "-",
        "--user-agent",
        "Teaching-Skill-Miner/1.2 full-video-research-download",
        "--output",
        str(partial),
        "--write-out",
        "%{url_effective}\n%{http_code}\n%{size_download}\n",
        url,
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=download_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        ensure_private_file(partial)
        raise RuntimeError(f"curl timed out while downloading {url}") from exc
    ensure_private_file(partial)
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise RuntimeError(f"curl failed for {url}: {detail or result.returncode}")
    receipt = parse_curl_write_out(result.stdout)
    receipt["resumed_from_bytes"] = resumed_from_bytes
    receipt["canonical_requested_url"] = url
    return receipt


def probe_local_media(
    media_path: str | Path,
    *,
    ffprobe_command: str = "ffprobe",
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    if timeout_seconds < 1:
        raise ValueError("ffprobe timeout must be positive")
    executable = _resolve_executable(ffprobe_command, label="ffprobe")
    path = Path(media_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"media path is missing or unsafe: {path}")
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                (
                    "format=duration,format_name,size:"
                    "stream=index,codec_type,codec_name,width,height,"
                    "avg_frame_rate,sample_rate,channels"
                ),
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out for {path}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ffprobe failed for {path}: {detail}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"ffprobe returned an invalid object for {path}")
    return value


def validate_media_probe(
    probe: dict[str, Any],
    *,
    reference_duration_seconds: float,
    actual_file_size_bytes: int | None = None,
    absolute_duration_tolerance_seconds: float = 2.0,
    relative_duration_tolerance: float = 0.001,
) -> dict[str, Any]:
    """Purely validate ffprobe output and return a privacy-minimal summary."""

    if not isinstance(probe, dict):
        raise ValueError("ffprobe result must be an object")
    try:
        reference_duration = float(reference_duration_seconds)
        absolute_tolerance = float(absolute_duration_tolerance_seconds)
        relative_tolerance = float(relative_duration_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("media probe tolerances must be numeric") from exc
    if (
        not math.isfinite(reference_duration)
        or reference_duration <= 0
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
        or not math.isfinite(relative_tolerance)
        or relative_tolerance < 0
    ):
        raise ValueError("media probe duration or tolerances are invalid")
    format_value = probe.get("format")
    streams = probe.get("streams")
    if not isinstance(format_value, dict) or not isinstance(streams, list):
        raise ValueError("ffprobe result lacks format or streams")
    try:
        duration = float(format_value.get("duration"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ffprobe media duration is invalid") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("ffprobe media duration is invalid")
    duration_tolerance = max(
        absolute_tolerance, relative_tolerance * reference_duration
    )
    duration_delta = abs(duration - reference_duration)
    if duration_delta > duration_tolerance:
        raise ValueError(
            "downloaded media duration differs from the formal reference: "
            f"expected approximately {reference_duration:.3f}, got {duration:.3f}"
        )
    container_names = {
        item.strip().lower()
        for item in str(format_value.get("format_name", "")).split(",")
        if item.strip()
    }
    if not container_names.intersection(SUPPORTED_CONTAINER_NAMES):
        raise ValueError("downloaded media is not an ffprobe-recognized MP4 container")
    if actual_file_size_bytes is not None:
        if actual_file_size_bytes <= 0:
            raise ValueError("downloaded media file is empty")
        probe_size = format_value.get("size")
        if probe_size is not None:
            try:
                parsed_probe_size = int(probe_size)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("ffprobe media size is invalid") from exc
            if parsed_probe_size != actual_file_size_bytes:
                raise ValueError("ffprobe media size differs from the local file size")

    video_streams: list[dict[str, Any]] = []
    audio_streams: list[dict[str, Any]] = []
    for index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            raise ValueError(f"ffprobe stream {index} is not an object")
        stream_type = stream.get("codec_type")
        if stream_type == "video":
            summary = {
                key: stream[key]
                for key in (
                    "index",
                    "codec_name",
                    "width",
                    "height",
                    "avg_frame_rate",
                )
                if key in stream
            }
            video_streams.append(summary)
        elif stream_type == "audio":
            summary = {
                key: stream[key]
                for key in ("index", "codec_name", "sample_rate", "channels")
                if key in stream
            }
            audio_streams.append(summary)
    if not video_streams:
        raise ValueError("downloaded media has no video stream")
    if not audio_streams:
        raise ValueError("downloaded media has no audio stream")
    return {
        "duration_seconds": round(duration, 6),
        "reference_duration_seconds": round(reference_duration, 6),
        "duration_delta_seconds": round(duration - reference_duration, 6),
        "duration_tolerance_seconds": round(duration_tolerance, 6),
        "container_names": sorted(container_names),
        "video_streams": video_streams,
        "audio_streams": audio_streams,
    }


def _existing_manifest_records(
    manifest_path: Path,
    *,
    bindings: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("existing media manifest target is unsafe")
    previous = read_json(manifest_path)
    if not isinstance(previous, dict):
        raise ValueError("existing media manifest is not an object")
    for field in ("source_manifest_sha256", "formal_caption_manifest_sha256"):
        if previous.get(field) != bindings[field]:
            raise ValueError(
                "existing media manifest is bound to different formal source data"
            )
    records = previous.get("videos", [])
    if not isinstance(records, list):
        raise ValueError("existing media manifest videos must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("video_id"), str):
            raise ValueError("existing media manifest contains an invalid record")
        if item["video_id"] in by_id:
            raise ValueError("existing media manifest contains duplicate video IDs")
        by_id[item["video_id"]] = item
    return by_id


def download_full_video_dataset(
    source_manifest_path: str | Path,
    formal_caption_manifest_path: str | Path,
    output_directory: str | Path,
    *,
    acknowledge_source_terms: bool = False,
    expected_video_count: int = 10,
    curl_command: str = "curl",
    ffprobe_command: str = "ffprobe",
    connect_timeout_seconds: int = 30,
    download_timeout_seconds: int = 14_400,
    ffprobe_timeout_seconds: int = 180,
    absolute_duration_tolerance_seconds: float = 2.0,
    relative_duration_tolerance: float = 0.001,
    downloader: Callable[..., dict[str, Any]] = curl_download_to_partial,
    media_probe: Callable[..., dict[str, Any]] = probe_local_media,
    generated_at_utc: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download and validate the complete formal-caption-linked video set.

    No upstream media hashes are asserted: each completed file receives a local
    post-download SHA-256 that detects later mutation but does not independently
    authenticate the publisher's bytes.
    """

    if not acknowledge_source_terms:
        raise ValueError(
            "full video download requires acknowledge_source_terms=True; MIT OCW "
            "media retains its upstream license and terms"
        )
    plan, bindings = load_full_video_plan(
        source_manifest_path,
        formal_caption_manifest_path,
        expected_video_count=expected_video_count,
    )
    generated_time = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    output = ensure_private_directory(output_directory)
    videos_directory = ensure_private_directory(output / "videos")
    manifest_path = output / "media_manifest.json"
    previous_records = _existing_manifest_records(manifest_path, bindings=bindings)
    records: list[dict[str, Any]] = []
    downloaded_count = 0
    reused_count = 0
    for item in plan:
        video_id = item["video_id"]
        final_path = videos_directory / f"{video_id}.mp4"
        partial_path = videos_directory / f"{video_id}.mp4.partial"
        if final_path.is_symlink() or (final_path.exists() and not final_path.is_file()):
            raise ValueError(f"final video target is unsafe: {final_path}")
        previous = previous_records.get(video_id, {})
        download_receipt: dict[str, Any]
        if final_path.exists():
            ensure_private_file(final_path)
            digest = file_sha256(final_path)
            previous_digest = previous.get("media_sha256")
            if previous_digest and previous_digest != digest:
                raise ValueError(
                    f"existing media SHA-256 differs from its manifest: {video_id}"
                )
            status = "reused_verified_local_file"
            reused_count += 1
            effective_url = previous.get("effective_download_url", item["media_url"])
            require_archive_https_url(effective_url, allow_cdn=True)
            download_receipt = {
                "effective_url": effective_url,
                "http_status": None,
                "downloaded_bytes_this_attempt": 0,
                "resumed_from_bytes": 0,
            }
        else:
            if partial_path.is_symlink() or (
                partial_path.exists() and not partial_path.is_file()
            ):
                raise ValueError(f"partial video target is unsafe: {partial_path}")
            download_receipt = downloader(
                item["media_url"],
                partial_path,
                curl_command=curl_command,
                connect_timeout_seconds=connect_timeout_seconds,
                download_timeout_seconds=download_timeout_seconds,
            )
            if not isinstance(download_receipt, dict):
                raise RuntimeError("video downloader returned an invalid receipt")
            effective_url = require_archive_https_url(
                download_receipt.get("effective_url"), allow_cdn=True
            )
            if partial_path.is_symlink() or not partial_path.is_file():
                raise RuntimeError(f"video downloader did not create {partial_path}")
            ensure_private_file(partial_path)
            if partial_path.stat().st_size <= 0:
                raise RuntimeError(f"video downloader created an empty file: {video_id}")
            with partial_path.open("rb") as handle:
                os.fsync(handle.fileno())
            status = "downloaded_and_validated"
            downloaded_count += 1

        media_path_for_probe = final_path if final_path.exists() else partial_path
        file_size = media_path_for_probe.stat().st_size
        probe = media_probe(
            media_path_for_probe,
            ffprobe_command=ffprobe_command,
            timeout_seconds=ffprobe_timeout_seconds,
        )
        probe_summary = validate_media_probe(
            probe,
            reference_duration_seconds=item["reference_media_duration_seconds"],
            actual_file_size_bytes=file_size,
            absolute_duration_tolerance_seconds=(
                absolute_duration_tolerance_seconds
            ),
            relative_duration_tolerance=relative_duration_tolerance,
        )
        digest = file_sha256(media_path_for_probe)
        if media_path_for_probe == partial_path:
            os.replace(partial_path, final_path)
            ensure_private_file(final_path)
        record = {
            "video_id": video_id,
            "course_id": item["course_id"],
            "title": item["title"],
            "source_url": item["source_url"],
            "requested_media_url": item["media_url"],
            "effective_download_url": effective_url,
            "media_path": f"videos/{video_id}.mp4",
            "media_size_bytes": file_size,
            "media_sha256": digest,
            "upstream_media_sha256_pinned": False,
            "formal_caption_sha256": item["caption_sha256"],
            "formal_transcript_sha256": item["formal_transcript_sha256"],
            "formal_transcript_path": item["formal_transcript_path"],
            "media_probe": probe_summary,
            "status": status,
            "download_transport": {
                "https_only": True,
                "archive_org_host_restricted": True,
                "http_status": download_receipt.get("http_status"),
                "downloaded_bytes_this_attempt": download_receipt.get(
                    "downloaded_bytes_this_attempt", 0
                ),
                "resumed_from_bytes": download_receipt.get(
                    "resumed_from_bytes", 0
                ),
            },
        }
        records.append(record)
        if progress:
            progress(
                f"{video_id}: {status}, {file_size} bytes, "
                f"duration {probe_summary['duration_seconds']:.3f}s"
            )

    manifest = {
        "artifact_kind": "private_full_video_dataset_manifest",
        "schema_version": "1.0",
        "generated_at_utc": generated_time,
        "dataset_id": bindings["dataset_id"],
        "publisher": bindings["publisher"],
        "license_url": bindings["license_url"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "formal_caption_manifest_sha256": bindings[
            "formal_caption_manifest_sha256"
        ],
        "source_terms_acknowledged": True,
        "storage_scope": "private_local_research_artifact",
        "raw_media_publicly_exported": False,
        "upstream_media_hashes_available": False,
        "integrity_interpretation": (
            "Local SHA-256 values detect mutation after HTTPS download. Because "
            "the source index does not pin publisher-provided media hashes, they "
            "do not independently authenticate upstream bytes. Duration and "
            "audio/video streams are checked with ffprobe against formal-caption "
            "references."
        ),
        "video_count": len(records),
        "complete": len(records) == expected_video_count,
        "videos": records,
    }
    receipt = {
        "artifact_kind": "private_full_video_download_receipt",
        "schema_version": "1.0",
        "generated_at_utc": generated_time,
        "dataset_id": bindings["dataset_id"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "formal_caption_manifest_sha256": bindings[
            "formal_caption_manifest_sha256"
        ],
        "source_terms_acknowledged": True,
        "complete": manifest["complete"],
        "video_count": len(records),
        "downloaded_count": downloaded_count,
        "reused_verified_count": reused_count,
        "total_media_bytes": sum(item["media_size_bytes"] for item in records),
        "media_content_included": False,
        "caption_text_included": False,
        "public_artifact": False,
        "publisher_media_hashes_pinned": False,
        "media_manifest_canonical_sha256": _canonical_sha256(manifest),
        "records": [
            {
                "video_id": item["video_id"],
                "media_sha256": item["media_sha256"],
                "media_size_bytes": item["media_size_bytes"],
                "duration_seconds": item["media_probe"]["duration_seconds"],
                "status": item["status"],
            }
            for item in records
        ],
    }
    write_json(manifest_path, manifest)
    receipt_path = write_json(output / "download_receipt.json", receipt)
    return {
        "output_directory": str(output.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "manifest": manifest,
        "receipt": receipt,
    }


def build_public_full_video_receipt(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a text/frame-free receipt for an already validated private manifest."""

    videos = manifest.get("videos", [])
    if not manifest.get("complete") or not isinstance(videos, list) or not videos:
        raise ValueError("a complete private full-video manifest is required")
    records = []
    for item in videos:
        digest = str(item.get("media_sha256", ""))
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("private media manifest contains an invalid SHA-256")
        records.append(
            {
                "video_id": item["video_id"],
                "course_id": item["course_id"],
                "media_size_bytes": item["media_size_bytes"],
                "media_sha256": digest,
                "duration_seconds": item["media_probe"]["duration_seconds"],
                "video_stream_count": len(
                    item["media_probe"].get("video_streams", [])
                ),
                "audio_stream_count": len(
                    item["media_probe"].get("audio_streams", [])
                ),
            }
        )
    return {
        "artifact_kind": "public_full_video_validation_receipt",
        "schema_version": "1.0",
        "generated_at_utc": manifest.get("generated_at_utc"),
        "dataset_id": manifest.get("dataset_id"),
        "publisher": manifest.get("publisher"),
        "license_url": manifest.get("license_url"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "formal_caption_manifest_sha256": manifest.get(
            "formal_caption_manifest_sha256"
        ),
        "complete": True,
        "video_count": len(records),
        "total_media_bytes": sum(item["media_size_bytes"] for item in records),
        "total_duration_seconds": round(
            sum(float(item["duration_seconds"]) for item in records), 3
        ),
        "media_content_included": False,
        "caption_text_included": False,
        "frame_content_included": False,
        "local_file_paths_included": False,
        "upstream_media_hashes_available": False,
        "integrity_interpretation": manifest.get("integrity_interpretation"),
        "records": records,
    }
