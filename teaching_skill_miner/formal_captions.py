from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .audit import audit_dataset
from .io_utils import ensure_private_directory, ensure_private_file, write_json
from .models import validate_transcript
from .preprocess import parse_srt_or_vtt


MIT_OCW_HOST = "ocw.mit.edu"
ARCHIVE_MEDIA_HOSTS = {"archive.org", "www.archive.org"}
MIT_OCW_CAPTION_LICENSE_URL = (
    "https://creativecommons.org/licenses/by-nc-sa/4.0/"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_HTML_BYTES = 5 * 1024 * 1024
MAX_CAPTION_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class DownloadedResource:
    requested_url: str
    final_url: str
    content_type: str
    body: bytes

    @property
    def sha256(self) -> str:
        return sha256(self.body).hexdigest()


@dataclass(frozen=True)
class VideoDeclaration:
    media_url: str
    caption_urls: tuple[str, ...]


class _VideoPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._current_media_url: str | None = None
        self._current_caption_urls: list[str] = []
        self.declarations: list[VideoDeclaration] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {key.lower(): value for key, value in attrs}
        if tag.lower() == "video":
            self._current_media_url = values.get("data-downloadlink")
            self._current_caption_urls = []
        elif tag.lower() == "track" and self._current_media_url:
            if values.get("kind", "").lower() != "captions":
                return
            language = (values.get("srclang") or "").lower()
            if language and not language.startswith("en"):
                return
            source = values.get("src")
            if source:
                self._current_caption_urls.append(source)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "video" or not self._current_media_url:
            return
        self.declarations.append(
            VideoDeclaration(
                media_url=self._current_media_url,
                caption_urls=tuple(self._current_caption_urls),
            )
        )
        self._current_media_url = None
        self._current_caption_urls = []


def _require_https_mit_url(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty URL")
    url = value.strip()
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != MIT_OCW_HOST
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"{field} must use https://{MIT_OCW_HOST}")
    return url


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _download_http(
    url: str,
    *,
    timeout_seconds: int,
    maximum_bytes: int,
) -> DownloadedResource:
    request = Request(
        url,
        headers={
            "User-Agent": "Teaching-Skill-Miner/1.2 formal-caption-audit",
            "Accept": "text/html,text/vtt,text/plain;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = -1
                if declared_size > maximum_bytes:
                    raise ValueError(
                        f"remote resource exceeds {maximum_bytes} bytes: {url}"
                    )
            body = response.read(maximum_bytes + 1)
            if len(body) > maximum_bytes:
                raise ValueError(
                    f"remote resource exceeds {maximum_bytes} bytes: {url}"
                )
            return DownloadedResource(
                requested_url=url,
                final_url=response.geturl(),
                content_type=response.headers.get_content_type(),
                body=body,
            )
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not download {url}: {exc}") from exc


def _extract_official_video_declaration(
    page_html: bytes,
    *,
    page_url: str,
    expected_caption_url: str,
    expected_media_url: str,
) -> VideoDeclaration:
    try:
        html_text = page_html.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"official source page is not UTF-8: {page_url}") from exc
    parser = _VideoPageParser()
    parser.feed(html_text)
    expected_caption = urljoin(page_url, expected_caption_url)
    expected_media = urljoin(page_url, expected_media_url)
    matches = []
    for declaration in parser.declarations:
        media = urljoin(page_url, declaration.media_url)
        captions = tuple(urljoin(page_url, item) for item in declaration.caption_urls)
        if media == expected_media and expected_caption in captions:
            matches.append(VideoDeclaration(media_url=media, caption_urls=captions))
    if len(matches) != 1:
        raise ValueError(
            "official page must declare exactly one matching video/caption pair: "
            f"{page_url}"
        )
    return matches[0]


def _media_probe_url(page_declared_url: str) -> str:
    parsed = urlparse(page_declared_url)
    if parsed.hostname not in ARCHIVE_MEDIA_HOSTS:
        raise ValueError(
            "MIT OCW page media must resolve to an approved Internet Archive host"
        )
    if parsed.username or parsed.password or parsed.scheme not in {"http", "https"}:
        raise ValueError("official media URL is unsafe")
    if parsed.scheme == "http":
        return parsed._replace(scheme="https", netloc=parsed.netloc).geturl()
    return page_declared_url


def _ffprobe_version(command: str) -> str:
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError(f"formal caption import requires `{command}` on PATH")
    try:
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"could not record {command} version") from exc
    lines = (result.stdout or result.stderr).splitlines()
    if result.returncode != 0 or not lines:
        raise RuntimeError(f"could not record {command} version")
    return lines[0].strip()


def _probe_media_duration(
    media_url: str,
    *,
    ffprobe_command: str,
    timeout_seconds: int,
) -> float:
    executable = shutil.which(ffprobe_command)
    if not executable:
        raise RuntimeError(
            f"formal caption import requires `{ffprobe_command}` on PATH"
        )
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                media_url,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out for {media_url}") from exc
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError, OverflowError) as exc:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ffprobe did not return a duration: {detail}") from exc
    if result.returncode != 0 or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"ffprobe returned an invalid duration for {media_url}")
    return duration


def _atomic_write_bytes(path: Path, body: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return ensure_private_file(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_source_manifest(source: dict[str, Any]) -> list[dict[str, Any]]:
    if source.get("source_kind") != "mit_ocw_official_caption_index":
        raise ValueError("unsupported formal caption source manifest kind")
    if source.get("publisher") != "MIT OpenCourseWare":
        raise ValueError("formal caption publisher must be MIT OpenCourseWare")
    _require_https_mit_url(source.get("publisher_url"), field="publisher_url")
    license_url = source.get("license_url")
    if license_url != MIT_OCW_CAPTION_LICENSE_URL:
        raise ValueError(
            "formal caption source manifest must record the verified MIT OCW "
            "CC BY-NC-SA 4.0 caption license"
        )
    videos = source.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("formal caption source manifest has no videos")
    required = {
        "video_id",
        "course_id",
        "title",
        "source_url",
        "caption_url",
        "media_url",
        "caption_sha256",
        "reference_media_duration_seconds",
        "reference_last_caption_cue_end_seconds",
    }
    seen_ids: set[str] = set()
    seen_pages: set[str] = set()
    seen_captions: set[str] = set()
    seen_hashes: set[str] = set()
    for index, item in enumerate(videos):
        if not isinstance(item, dict):
            raise ValueError(f"videos[{index}] must be an object")
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"videos[{index}] missing: {', '.join(missing)}")
        for field in ("video_id", "course_id", "title"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"videos[{index}].{field} must be non-empty")
        page = _require_https_mit_url(
            item.get("source_url"), field=f"videos[{index}].source_url"
        )
        caption = _require_https_mit_url(
            item.get("caption_url"), field=f"videos[{index}].caption_url"
        )
        digest = _require_sha256(
            item.get("caption_sha256"), field=f"videos[{index}].caption_sha256"
        )
        media = str(item.get("media_url", "")).strip()
        _media_probe_url(media)
        for field in (
            "reference_media_duration_seconds",
            "reference_last_caption_cue_end_seconds",
        ):
            try:
                reference_value = float(item.get(field))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"videos[{index}].{field} must be numeric") from exc
            if not math.isfinite(reference_value) or reference_value <= 0:
                raise ValueError(f"videos[{index}].{field} must be positive")
        video_id = item["video_id"]
        if video_id in seen_ids:
            raise ValueError(f"duplicate video_id: {video_id}")
        if page in seen_pages:
            raise ValueError(f"duplicate source_url: {page}")
        if caption in seen_captions:
            raise ValueError(f"duplicate caption_url: {caption}")
        if digest in seen_hashes:
            raise ValueError(f"duplicate caption_sha256: {digest}")
        seen_ids.add(video_id)
        seen_pages.add(page)
        seen_captions.add(caption)
        seen_hashes.add(digest)
    return videos


def import_mit_ocw_formal_captions(
    source_manifest_path: str | Path,
    output_directory: str | Path,
    *,
    ffprobe_command: str = "ffprobe",
    timeout_seconds: int = 120,
    downloader: Callable[..., DownloadedResource] = _download_http,
    media_duration_probe: Callable[..., float] = _probe_media_duration,
    retrieved_at_utc: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch hash-pinned MIT OCW captions and build a private formal dataset.

    Official-source status is asserted only after the known OCW page declares the
    exact caption/media pair and the downloaded VTT matches its pinned SHA-256.
    """

    source_path = Path(source_manifest_path)
    raw_source = source_path.read_bytes()
    try:
        source = json.loads(raw_source.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal caption source manifest is not valid UTF-8 JSON") from exc
    if not isinstance(source, dict):
        raise ValueError("formal caption source manifest must be an object")
    videos = _validate_source_manifest(source)
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    output = ensure_private_directory(output_directory)
    retrieval_time = retrieved_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    ffprobe_version = _ffprobe_version(ffprobe_command)
    transcript_payloads: list[tuple[dict[str, Any], bytes, dict[str, Any]]] = []
    for item in videos:
        video_id = item["video_id"]
        page_url = _require_https_mit_url(item["source_url"], field="source_url")
        caption_url = _require_https_mit_url(
            item["caption_url"], field="caption_url"
        )
        expected_caption_sha256 = _require_sha256(
            item["caption_sha256"], field="caption_sha256"
        )
        expected_media_url = str(item["media_url"]).strip()
        page = downloader(
            page_url,
            timeout_seconds=timeout_seconds,
            maximum_bytes=MAX_HTML_BYTES,
        )
        _require_https_mit_url(page.final_url, field="final source page URL")
        if page.content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError(f"official source page is not HTML: {page_url}")
        declaration = _extract_official_video_declaration(
            page.body,
            page_url=page_url,
            expected_caption_url=caption_url,
            expected_media_url=expected_media_url,
        )
        caption = downloader(
            caption_url,
            timeout_seconds=timeout_seconds,
            maximum_bytes=MAX_CAPTION_BYTES,
        )
        _require_https_mit_url(caption.final_url, field="final caption URL")
        if caption.content_type != "text/vtt":
            raise ValueError(
                f"official caption content type is not text/vtt: {video_id}"
            )
        if caption.sha256 != expected_caption_sha256:
            raise ValueError(
                f"caption SHA-256 mismatch for {video_id}: "
                f"expected {expected_caption_sha256}, got {caption.sha256}"
            )
        try:
            caption_text = caption.body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"caption is not UTF-8: {video_id}") from exc
        if not caption_text.lstrip().startswith("WEBVTT"):
            raise ValueError(f"official caption is not WebVTT: {video_id}")
        segments = parse_srt_or_vtt(caption_text)
        if not segments:
            raise ValueError(f"official caption has no timed cues: {video_id}")
        media_url = _media_probe_url(declaration.media_url)
        source_duration = media_duration_probe(
            media_url,
            ffprobe_command=ffprobe_command,
            timeout_seconds=timeout_seconds,
        )
        first_cue_start = min(float(segment["start"]) for segment in segments)
        last_cue_end = max(float(segment["end"]) for segment in segments)
        covered_duration = last_cue_end - first_cue_start
        reference_duration = float(item["reference_media_duration_seconds"])
        duration_tolerance = max(2.0, 0.001 * reference_duration)
        if abs(source_duration - reference_duration) > duration_tolerance:
            raise ValueError(
                f"media duration changed for {video_id}: expected approximately "
                f"{reference_duration:.3f}, got {source_duration:.3f}"
            )
        reference_caption_end = float(
            item["reference_last_caption_cue_end_seconds"]
        )
        if abs(last_cue_end - reference_caption_end) > 0.01:
            raise ValueError(
                f"caption endpoint changed for {video_id}: expected "
                f"{reference_caption_end:.3f}, got {last_cue_end:.3f}"
            )
        coverage_fraction = min(1.0, covered_duration / source_duration)
        complete = bool(
            coverage_fraction >= 0.95
            and last_cue_end <= source_duration + 5.0
        )
        if not complete:
            raise ValueError(
                f"caption timeline-span coverage is insufficient for {video_id}: "
                f"{coverage_fraction:.4%}"
            )
        transcript = {
            "video_id": video_id,
            "course_id": item["course_id"],
            "title": item["title"],
            "source_url": page_url,
            "transcript_url": caption_url,
            "language": item.get("language", "en"),
            "transcript_kind": "caption_import",
            "timestamps_are_approximate": False,
            "provenance": {
                "input_filename": Path(urlparse(caption_url).path).name,
                "input_sha256": caption.sha256,
                "input_size_bytes": len(caption.body),
                "pipeline": "teaching_skill_miner.formal_captions.v1",
                "timestamp_source": "source_caption",
                "retrieved_at_utc": retrieval_time,
                "caption_source_verified": True,
                "caption_source_verification": {
                    "publisher": source.get("publisher"),
                    "trusted_source_host": MIT_OCW_HOST,
                    "official_page_url": page_url,
                    "official_page_final_url": page.final_url,
                    "official_page_sha256": page.sha256,
                    "page_declared_caption_url": caption_url,
                    "caption_final_url": caption.final_url,
                    "caption_content_type": caption.content_type,
                    "caption_sha256_pinned_in_source_manifest": True,
                    "page_declared_media_url": declaration.media_url,
                    "media_probe_url": media_url,
                },
                "transcript_coverage": {
                    "source_duration_seconds": round(source_duration, 3),
                    "covered_duration_seconds": round(covered_duration, 3),
                    "coverage_fraction": round(coverage_fraction, 6),
                    "completeness_verified": True,
                    "verification_method": (
                        "official_mit_ocw_page_video_track_plus_"
                        "webvtt_endpoint_to_remote_media_ffprobe"
                    ),
                    "coverage_metric": (
                        "(last_caption_cue_end-first_caption_cue_start)/"
                        "media_duration"
                    ),
                    "ffprobe_version": ffprobe_version,
                },
            },
            "segments": segments,
        }
        validation = validate_transcript(transcript)
        if not validation.valid:
            raise ValueError(
                f"generated transcript is invalid for {video_id}: "
                + "; ".join(validation.errors)
            )
        receipt = {
            "video_id": video_id,
            "course_id": item["course_id"],
            "source_url": page_url,
            "caption_url": caption_url,
            "caption_sha256": caption.sha256,
            "caption_size_bytes": len(caption.body),
            "official_page_sha256": page.sha256,
            "media_url": media_url,
            "media_duration_seconds": round(source_duration, 3),
            "first_caption_cue_start_seconds": round(first_cue_start, 3),
            "last_caption_cue_end_seconds": round(last_cue_end, 3),
            "caption_timeline_span_seconds": round(covered_duration, 3),
            "timeline_span_coverage_fraction": round(coverage_fraction, 6),
            "segment_count": len(segments),
            "transcript_canonical_sha256": _canonical_sha256(transcript),
        }
        transcript_payloads.append((transcript, caption.body, receipt))
        if progress:
            progress(
                f"{video_id}: {len(segments)} cues, "
                f"timeline-span coverage {coverage_fraction:.2%}"
            )

    manifest = {
        "dataset_id": source.get("dataset_id"),
        "version": source.get("version", "1.0"),
        "created_for": "formal transcript provenance and completeness audit",
        "generated_at_utc": retrieval_time,
        "source_manifest_sha256": sha256(raw_source).hexdigest(),
        "publisher": source.get("publisher"),
        "publisher_url": source.get("publisher_url"),
        "license_url": source.get("license_url"),
        "license_note": (
            "Official captions retain their upstream license and are not "
            "relicensed by Teaching Skill Miner. Keep full text private or "
            "redistribute only under the upstream terms."
        ),
        "courses": source.get("courses", []),
        "videos": [
            {
                "video_id": transcript["video_id"],
                "course_id": transcript["course_id"],
                "title": transcript["title"],
                "source_url": transcript["source_url"],
                "transcript_path": f"transcripts/{transcript['video_id']}.json",
            }
            for transcript, _, _ in transcript_payloads
        ],
    }
    transcripts_by_id = {
        item[0]["video_id"]: item[0] for item in transcript_payloads
    }
    with tempfile.TemporaryDirectory(prefix="tsm-formal-caption-audit-") as staging:
        staging_root = Path(staging)
        for transcript, _, _ in transcript_payloads:
            write_json(
                staging_root / "transcripts" / f"{transcript['video_id']}.json",
                transcript,
            )
        write_json(staging_root / "dataset_manifest.json", manifest)
        audit = audit_dataset(manifest, staging_root)
    if not audit["formal_empirical_ready"]:
        raise RuntimeError(
            "generated formal caption dataset failed its own audit: "
            + "; ".join(audit.get("errors", []))
        )
    raw_output = ensure_private_directory(output / "raw")
    transcript_output = ensure_private_directory(output / "transcripts")
    for transcript, caption_body, _ in transcript_payloads:
        video_id = transcript["video_id"]
        _atomic_write_bytes(raw_output / f"{video_id}.vtt", caption_body)
        write_json(transcript_output / f"{video_id}.json", transcript)
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "data_audit.json", audit)
    receipts = [item[2] for item in transcript_payloads]
    public_receipt = {
        "artifact_kind": "formal_caption_retrieval_receipt",
        "schema_version": "1.0",
        "generated_at_utc": retrieval_time,
        "dataset_id": manifest["dataset_id"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "formal_empirical_ready": audit["formal_empirical_ready"],
        "video_count": audit["video_count"],
        "research_grade_transcript_count": audit[
            "research_grade_transcript_count"
        ],
        "caption_text_included": False,
        "raw_media_downloaded": False,
        "coverage_interpretation": (
            "timeline-span coverage verifies that the official caption track "
            "runs from its first cue to near the end of the official page-linked "
            "media; it is not frame-level speech coverage or an ASR WER measurement"
        ),
        "ffprobe_version": ffprobe_version,
        "license_url": source.get("license_url"),
        "records": receipts,
        "dataset_manifest_canonical_sha256": _canonical_sha256(manifest),
        "transcript_set_sha256": _canonical_sha256(transcripts_by_id),
    }
    write_json(output / "retrieval_receipt.json", public_receipt)
    return {
        "output_directory": str(output.resolve()),
        "manifest_path": str((output / "dataset_manifest.json").resolve()),
        "audit_path": str((output / "data_audit.json").resolve()),
        "receipt_path": str((output / "retrieval_receipt.json").resolve()),
        "audit": audit,
        "receipt": public_receipt,
    }
